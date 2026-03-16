--[[
SSH-android plugin for KOReader
Mimics simplesshd bootstrap auth behavior.

=== Why the patched dropbear generates the bootstrap password ===
Android has no /etc/shadow and no root-accessible PAM. The only way to inject
a password into dropbear's auth flow without root is inside the auth code
itself. The simplesshd patches (svr-auth.c) do exactly this: they replace the
standard auth-type selection with a runtime check:
  - authorized_keys present and >= 10 bytes → pubkey-only auth
  - otherwise → generate 8 random chars via genrandom(), store in
    ses.authstate.pw_passwd, advertise AUTH_TYPE_PASSWORD
The generated password is logged to stderr (captured to dropbear.err) so it
can be shown to the user. This is the same approach the simplesshd Android app
takes (it reads dropbear.err and shows the password in a notification).

=== scp / rsync ===
The patched dropbear (dbutil.c) intercepts remote commands:
  "scp ..."   → exec  $SSHD_LIBDIR/libscp.so   <args>
  "rsync ..." → exec  $SSHD_LIBDIR/librsync.so <args>
$SSHD_LIBDIR is set inside dropbear before exec'ing the user shell. It must be
initialised at dropbear startup from the SSHD_LIBDIR environment variable we
set before launching the binary (our standalone build reads this env var to
populate the conf_lib C global, since we don't have the JNI glue of the
original app).

=== Android binary execution (Android 10+ W^X policy) ===
Binaries are packaged as .so files (libdropbear.so, libscp.so, librsync.so) in
jniLibs/ so they land in nativeLibraryDir, which is exec-capable. The plugin
finds them via android.nativeLibraryDir at runtime.

=== Paths used by the patched dropbear ===
  SSHD_CONF_PATH → the conf directory; dropbear looks for:
    authorized_keys          (pubkey auth)
    dropbear_rsa_host_key    (host key, generated on first connect with -R)
    dropbear_ecdsa_host_key
    dropbear_ed25519_host_key
    dropbear.pid             (PID file, matches DROPBEAR_PIDFILE default)
    dropbear.err             (stderr, where the bootstrap password is logged)
  SSHD_LIBDIR    → directory containing libscp.so / librsync.so
--]]

local BD = require("ui/bidi")
local DataStorage = require("datastorage")
local Device = require("device")
local Dispatcher = require("dispatcher")
local InfoMessage = require("ui/widget/infomessage")
local InputDialog = require("ui/widget/inputdialog")
local UIManager = require("ui/uimanager")
local WidgetContainer = require("ui/widget/container/widgetcontainer")
local ffiutil = require("ffi/util")
local logger = require("logger")
local util = require("util")
local _ = require("gettext")
local T = ffiutil.template

local hasAndroid, android = pcall(require, "android")

local ffi = require("ffi")
pcall(function()
    ffi.cdef([[
        int    setenv(const char *name, const char *value, int overwrite);
        int    unsetenv(const char *name);
        pid_t  fork(void);
        int    execv(const char *path, char *const argv[]);
        void   _exit(int status);
        pid_t  waitpid(pid_t pid, int *status, int options);
    ]])
end)

-- Launch a binary asynchronously without blocking the calling thread.
--
-- os.execute() calls system() which calls waitpid() — it blocks until the
-- shell and everything it spawns has exited. On Android this freezes the UI.
-- We bypass it with a double-fork so the Lua thread never waits:
--
--   KOReader (A) → fork → child (B) → fork → grandchild (C)
--                          B exits immediately              ↓
--   A: waitpid(B) returns in microseconds       C: exec(binary)
--                                               C is orphaned to init,
--                                               never zombies in A.
--
-- env_pairs is a list of {name, value} pairs set in A before forking (and
-- unset in A afterwards); the forks inherit them and exec passes them on.
local function spawnBackground(binary, args, env_pairs)
    for _, kv in ipairs(env_pairs or {}) do
        pcall(ffi.C.setenv, kv[1], kv[2], 1)
    end

    local child = ffi.C.fork()
    if child < 0 then
        for _, kv in ipairs(env_pairs or {}) do pcall(ffi.C.unsetenv, kv[1]) end
        return false, "fork() failed"
    end

    if child == 0 then
        -- First child (B): fork the real worker then exit immediately.
        local grandchild = ffi.C.fork()
        if grandchild ~= 0 then
            ffi.C._exit(0)  -- B exits; grandchild C is orphaned to init
        end
        -- Grandchild (C): exec the binary.
        local n = #args
        local argv = ffi.new("const char*[?]", n + 2)
        argv[0] = binary
        for i = 1, n do argv[i] = args[i] end
        argv[n + 1] = nil
        ffi.C.execv(binary, ffi.cast("char *const*", argv))
        ffi.C._exit(1)  -- execv failed
    end

    -- Parent (A): reap B immediately (B exits right after its own fork).
    ffi.C.waitpid(child, nil, 0)

    -- Unset env vars from A's environment now that the forks have inherited them.
    for _, kv in ipairs(env_pairs or {}) do pcall(ffi.C.unsetenv, kv[1]) end

    return true
end

-- ---------------------------------------------------------------------------
-- Paths
-- ---------------------------------------------------------------------------

local DATA_DIR  = DataStorage:getFullDataDir()
-- Persistent config (authorized_keys, host keys) — survives cache clears.
local CONF_DIR  = DataStorage:getDataDir() .. "/settings/SSH-android"
-- Runtime state (pid file, log) — goes in cache; safe to wipe.
-- standalone.c must be built with SSHD_RUN_PATH support for these to land here.
local RUN_DIR   = DataStorage:getDataDir() .. "/cache/SSH-android"
local PID_FILE  = RUN_DIR  .. "/dropbear.pid"
local LOG_FILE  = RUN_DIR  .. "/dropbear.err"
local AUTH_KEYS = CONF_DIR .. "/authorized_keys"

-- ---------------------------------------------------------------------------
-- Binary resolution
-- ---------------------------------------------------------------------------

-- On Android 10+, exec() from writable paths fails (W^X policy).
-- Binaries are packaged as .so files in jniLibs/, which the installer places
-- in nativeLibraryDir — an exec-capable, read-only path.
-- Fallback to a plain "dropbear" binary for non-Android platforms.
local function resolveDropbear()
    if hasAndroid and android and android.nativeLibraryDir then
        local p = android.nativeLibraryDir .. "/libdropbear.so"
        if util.pathExists(p) then
            return p, android.nativeLibraryDir
        end
    end
    if util.pathExists("dropbear") then
        return "./dropbear", nil
    end
    return nil, nil
end

-- ---------------------------------------------------------------------------
-- Plugin definition
-- ---------------------------------------------------------------------------

local SSHAndroid = WidgetContainer:extend{
    name = "SSH-android",
    is_doc_only = false,
}

function SSHAndroid:init()
    self.SSH_port = G_reader_settings:readSetting("SSH_android_port") or "2222"
    self.autostart = G_reader_settings:isTrue("SSH_android_autostart")

    local binary = resolveDropbear()
    if not binary then
        -- No dropbear binary available; plugin is a no-op.
        return
    end

    if self.autostart then
        self:start()
    end

    self.ui.menu:registerToMainMenu(self)
    self:onDispatcherRegisterActions()
end

-- ---------------------------------------------------------------------------
-- Auth helpers
-- ---------------------------------------------------------------------------

-- Mirrors the C authkeys_exists() check: file must exist and have >= 10 bytes.
-- (MIN_AUTHKEYS_LINE = 10 in the simplesshd dropbear patches.)
local function hasAuthorizedKeys()
    local f = io.open(AUTH_KEYS, "r")
    if not f then return false end
    local chunk = f:read(10)
    f:close()
    return chunk ~= nil and #chunk >= 10
end

-- Parse the bootstrap password from dropbear.err.
-- The patched dropbear (svr-auth.c) logs:
--   WARNING: no authorized keys, generating single-use password:
--   ALERT: --------
--   ALERT: <8-char password>
--   ALERT: --------
-- The exact prefix format depends on the dropbear log formatter; we strip
-- everything up to and including the last ": " on each relevant line.
local function readOneTimePassword()
    local f = io.open(LOG_FILE, "r")
    if not f then return nil end
    local content = f:read("*a")
    f:close()

    -- Quick exit if no password block in this log.
    -- NOTE: plain-mode find (3rd arg = true) treats the pattern as a literal
    -- string, so the search term must not use Lua pattern escapes like "%-".
    if not content:find("generating single-use password", 1, true) then
        return nil
    end

    local pw = nil
    local in_block = false
    local saw_open_dashes = false

    for line in content:gmatch("[^\n]+") do
        -- Strip "[PID] Mon DD HH:MM:SS " prefix emitted by the patched dropbear.
        -- The format has no colon after the timestamp, so a colon-based strip
        -- would eat into "21:21:19" and leave "19 message" instead of "message".
        local msg = line:match("%d+:%d+:%d+%s+(.-)%s*$") or line:gsub("^%s+", ""):gsub("%s+$", "")

        if line:find("generating single-use password", 1, true) then
            in_block = true
            saw_open_dashes = false
        elseif in_block then
            if msg == "--------" then
                if not saw_open_dashes then
                    saw_open_dashes = true
                else
                    -- Closing delimiter — end of this block.
                    in_block = false
                    saw_open_dashes = false
                end
            elseif saw_open_dashes and #msg >= 1 then
                -- The password is the first non-space token on this line.
                local candidate = msg:match("^(%S+)")
                if candidate and #candidate >= 8 then
                    pw = candidate:sub(1, 8)
                end
                in_block = false
                saw_open_dashes = false
            end
        end
    end
    return pw
end

-- Return {ipv4 = {...}, ipv6 = {...}} of non-loopback local IPs via `ip addr show`.
-- Skips addresses with "scope host" (loopback). Prefix length is stripped.
local function getLocalIPs()
    local ipv4, ipv6 = {}, {}
    local p = io.popen("ip addr show 2>/dev/null")
    if not p then return { ipv4 = ipv4, ipv6 = ipv6 } end
    for line in p:lines() do
        if not line:find("scope host", 1, true) then
            local ip4 = line:match("inet%s+(%d+%.%d+%.%d+%.%d+)/")
            if ip4 then ipv4[#ipv4 + 1] = ip4 end
            local ip6 = line:match("inet6%s+([^/%s]+)/")
            if ip6 then ipv6[#ipv6 + 1] = ip6 end
        end
    end
    p:close()
    return { ipv4 = ipv4, ipv6 = ipv6 }
end

-- Return a newline-joined string of all local IPs (v4 then v6), or a
-- connectivity description as fallback when `ip` is unavailable.
local function getNetworkInfo()
    local t = getLocalIPs()
    local lines = {}
    for _, a in ipairs(t.ipv4) do lines[#lines + 1] = a end
    for _, a in ipairs(t.ipv6) do lines[#lines + 1] = a end
    if #lines > 0 then return table.concat(lines, "\n") end
    return Device.retrieveNetworkInfo and Device:retrieveNetworkInfo()
        or _("Could not retrieve network info.")
end

-- ---------------------------------------------------------------------------
-- Start / stop
-- ---------------------------------------------------------------------------

function SSHAndroid:isRunning()
    if not util.pathExists(PID_FILE) then return false end
    -- Guard against stale PID files.
    local f = io.open(PID_FILE, "r")
    if not f then return false end
    local pid = tonumber(f:read("*l"))
    f:close()
    return pid ~= nil and util.pathExists("/proc/" .. pid)
end

function SSHAndroid:start()
    if self:isRunning() then
        logger.dbg("[SSH-android] Already running, skipping start.")
        return
    end

    local binary, lib_dir = resolveDropbear()
    if not binary then
        UIManager:show(InfoMessage:new{
            icon = "notice-warning",
            text = _("SSH server binary not found.\nCannot start SSH server."),
        })
        return
    end

    -- Ensure config and runtime directories exist.
    if not util.pathExists(CONF_DIR) then
        os.execute(string.format("mkdir -p %q", CONF_DIR))
    end
    if not util.pathExists(RUN_DIR) then
        os.execute(string.format("mkdir -p %q", RUN_DIR))
    end

    -- Note: log rotation (dropbear.err → dropbear.err.old) is handled inside
    -- the binary's standalone entry point (mirrors simplesshd interface.c).
    -- We do NOT rotate here to avoid a race with the binary's own rotation.

    -- Kindle firewall.
    if Device:isKindle() then
        os.execute(string.format(
            "iptables -A INPUT -p tcp --dport %s -m conntrack --ctstate NEW,ESTABLISHED -j ACCEPT",
            self.SSH_port))
        os.execute(string.format(
            "iptables -A OUTPUT -p tcp --sport %s -m conntrack --ctstate ESTABLISHED -j ACCEPT",
            self.SSH_port))
    end
    -- Kobo pseudo-terminals.
    if Device:isKobo() then
        os.execute([[if [ ! -d "/dev/pts" ]; then
            mkdir -p /dev/pts
            mount -t devpts devpts /dev/pts
            fi]])
    end

    -- Build env vars for the patched dropbear.
    --   SSHD_CONF_PATH: standalone.c reads this → conf_path → host keys,
    --                   pid file, and log file all land under CONF_DIR.
    --   SSHD_LIBDIR:    standalone.c reads this → conf_lib → dropbear finds
    --                   libscp.so/librsync.so when dispatching remote commands.
    --   LD_LIBRARY_PATH: ensures the dynamic linker finds shared-lib deps.
    -- spawnBackground() sets them before fork(), and unsets them in the parent
    -- after waitpid() returns — children have already inherited their own copies.
    local env = {
        { "SSHD_CONF_PATH", CONF_DIR },
        -- SSHD_RUN_PATH: standalone.c writes pid file and log here instead of
        -- CONF_DIR so runtime state stays in cache (safe to wipe).
        -- Requires standalone.c built with SSHD_RUN_PATH support.
        { "SSHD_RUN_PATH",  RUN_DIR  },
        -- SSHD_HOME: sets conf_home so the user shell lands in writable storage
        -- instead of the read-only /data/local default.
        -- Requires standalone.c built with SSHD_HOME support.
        { "SSHD_HOME",      DATA_DIR },
    }
    if lib_dir then
        env[#env + 1] = { "SSHD_LIBDIR",     lib_dir }
        env[#env + 1] = { "LD_LIBRARY_PATH", lib_dir }
    end

    -- -R : generate host keys on-demand (DROPBEAR_DELAY_HOSTKEY)
    -- -F : run in foreground; the grandchild is already orphaned to init by
    --      the double-fork, so it is fine for it to run as a foreground process.
    --      Without -F, dropbear would do an internal double-fork we don't need,
    --      and the daemonization pipe-wait could add latency.
    -- -p : port
    local db_args = { "-R", "-F", "-p", self.SSH_port }
    logger.dbg("[SSH-android] Launching:", binary, table.concat(db_args, " "))
    local ok, err = spawnBackground(binary, db_args, env)
    if not ok then
        logger.err("[SSH-android] spawnBackground failed:", err)
        UIManager:show(InfoMessage:new{
            icon = "notice-warning",
            text = _("Failed to start SSH server."),
        })
        return
    end

    -- Do NOT sleep here: this runs on the Android main/UI thread.
    -- The bootstrap password is not in the log yet anyway — it is generated by
    -- dropbear's svr_authinitialise() on the first client connection, not at
    -- server startup. Direct the user to "Show one-time password" instead.
    local net_info = getNetworkInfo()

    if hasAuthorizedKeys() then
        UIManager:show(InfoMessage:new{
            timeout = 10,
            text = T(_("SSH server started.\n\nPort: %1\nAuth: public key only\n%2"),
                self.SSH_port, net_info),
        })
    else
        UIManager:show(InfoMessage:new{
            timeout = 20,
            text = T(_("SSH server started (bootstrap mode).\n\nPort: %1\n%2\n\nConnect once via SSH to generate a one-time password, then tap 'Show one-time password'.\n\nTo add your public key after first login:\n  cat > %3\n  (paste key, then Ctrl+D)"),
                self.SSH_port,
                net_info,
                BD.filepath(AUTH_KEYS)),
        })
    end
end

function SSHAndroid:stop()
    if not self:isRunning() then
        return
    end

    local f = io.open(PID_FILE, "r")
    local pid = f and tonumber(f:read("*l"))
    if f then f:close() end

    local function alive(p)
        return p and util.pathExists("/proc/" .. p)
    end

    if pid then
        os.execute(string.format("kill -TERM %d", pid))
        for _ = 1, 20 do
            if not alive(pid) then break end
            ffiutil.sleep(0.1)
        end
        if alive(pid) then
            os.execute(string.format("kill -KILL %d", pid))
            for _ = 1, 10 do
                if not alive(pid) then break end
                ffiutil.sleep(0.1)
            end
        end
    end

    -- Plug Kindle firewall hole.
    if Device:isKindle() then
        os.execute(string.format(
            "iptables -D INPUT -p tcp --dport %s -m conntrack --ctstate NEW,ESTABLISHED -j ACCEPT",
            self.SSH_port))
        os.execute(string.format(
            "iptables -D OUTPUT -p tcp --sport %s -m conntrack --ctstate ESTABLISHED -j ACCEPT",
            self.SSH_port))
    end

    os.remove(PID_FILE)

    UIManager:show(InfoMessage:new{
        text = _("SSH server stopped."),
        timeout = 2,
    })
end

function SSHAndroid:onToggleSSHAndroidServer()
    if self:isRunning() then
        self:stop()
    else
        self:start()
    end
end

-- ---------------------------------------------------------------------------
-- Menu dialogs
-- ---------------------------------------------------------------------------

function SSHAndroid:show_port_dialog(touchmenu_instance)
    self.port_dialog = InputDialog:new{
        title = _("Choose SSH port"),
        input = self.SSH_port,
        input_type = "number",
        input_hint = self.SSH_port,
        buttons = {
            {
                {
                    text = _("Cancel"),
                    id = "close",
                    callback = function()
                        UIManager:close(self.port_dialog)
                    end,
                },
                {
                    text = _("Save"),
                    is_enter_default = true,
                    callback = function()
                        local value = tonumber(self.port_dialog:getInputText())
                        if value and value >= 1 and value <= 65535 then
                            self.SSH_port = tostring(value)
                            G_reader_settings:saveSetting("SSH_android_port", self.SSH_port)
                            UIManager:close(self.port_dialog)
                            touchmenu_instance:updateItems()
                        end
                    end,
                },
            },
        },
    }
    UIManager:show(self.port_dialog)
    self.port_dialog:onShowKeyboard()
end

function SSHAndroid:show_one_time_password()
    if not self:isRunning() then
        UIManager:show(InfoMessage:new{
            text = _("SSH server is not running."),
            timeout = 5,
        })
        return
    end
    if hasAuthorizedKeys() then
        UIManager:show(InfoMessage:new{
            text = T(_("authorized_keys found.\nPassword authentication is disabled.\n\nKey file: %1"),
                BD.filepath(AUTH_KEYS)),
            timeout = 10,
        })
        return
    end
    local pw = readOneTimePassword()
    if pw then
        UIManager:show(InfoMessage:new{
            text = T(_("One-time password for this session:\n\n%1\n\nTo add your public key (enables key-only auth):\n  cat > %2\n  (paste key, then Ctrl+D)"),
                pw, BD.filepath(AUTH_KEYS)),
            timeout = 60,
        })
    else
        -- The password is only generated when a client attempts to connect
        -- (dropbear's svr_authinitialise() runs per-connection, not at startup).
        UIManager:show(InfoMessage:new{
            text = T(_("No password in log yet.\n\nTry connecting via SSH first:\n  ssh -p %1 root@<device-ip>\n\nThen tap here again to see the generated password.\n\nLog: %2"),
                self.SSH_port,
                BD.filepath(LOG_FILE)),
            timeout = 30,
        })
    end
end

-- ---------------------------------------------------------------------------
-- Menu registration
-- ---------------------------------------------------------------------------

function SSHAndroid:addToMainMenu(menu_items)
    menu_items.ssh_android = {
        text         = _("SSH server (Android)"),
        sorting_hint = "more_tools",
        -- sub_item_table_func so text_func items re-evaluate on each open.
        sub_item_table_func = function()
            return {
                {
                    text = _("SSH server (Android)"),
                    checked_func = function() return self:isRunning() end,
                    callback = function(touchmenu_instance)
                        self:onToggleSSHAndroidServer()
                        ffiutil.sleep(1)
                        touchmenu_instance:updateItems()
                    end,
                },
                {
                    text_func = function()
                        return T(_("SSH port: %1"), self.SSH_port)
                    end,
                    keep_menu_open = true,
                    enabled_func = function() return not self:isRunning() end,
                    callback = function(touchmenu_instance)
                        self:show_port_dialog(touchmenu_instance)
                    end,
                },
                {
                    -- Auth status row; tap to show OTP detail.
                    text_func = function()
                        if hasAuthorizedKeys() then
                            return _("Auth: public key only")
                        else
                            return _("Auth: bootstrap (one-time password)")
                        end
                    end,
                    keep_menu_open = true,
                    callback = function()
                        self:show_one_time_password()
                    end,
                },
                {
                    text = _("Show IP addresses"),
                    keep_menu_open = true,
                    callback = function()
                        local t = getLocalIPs()
                        local lines = {}
                        if #t.ipv4 > 0 then
                            for _, a in ipairs(t.ipv4) do
                                lines[#lines + 1] = a
                            end
                        end
                        if #t.ipv6 > 0 then
                            if #lines > 0 then lines[#lines + 1] = "" end
                            for _, a in ipairs(t.ipv6) do
                                lines[#lines + 1] = a
                            end
                        end
                        local text = #lines > 0
                            and table.concat(lines, "\n")
                            or  _("No network addresses found.")
                        UIManager:show(InfoMessage:new{
                            text = text,
                            timeout = 30,
                        })
                    end,
                },
                {
                    text = _("Show one-time password"),
                    keep_menu_open = true,
                    enabled_func = function()
                        return self:isRunning() and not hasAuthorizedKeys()
                    end,
                    callback = function()
                        self:show_one_time_password()
                    end,
                },
                {
                    text = _("SSH public key location"),
                    keep_menu_open = true,
                    callback = function()
                        UIManager:show(InfoMessage:new{
                            timeout = 60,
                            text = T(_("Place your public SSH key in:\n%1\n\nOnce the file is present and non-empty, password authentication will be disabled automatically on the next server start."),
                                BD.filepath(AUTH_KEYS)),
                        })
                    end,
                },
                {
                    text = _("Start SSH server with KOReader"),
                    checked_func = function() return self.autostart end,
                    callback = function()
                        self.autostart = not self.autostart
                        G_reader_settings:flipNilOrFalse("SSH_android_autostart")
                    end,
                },
            }
        end,
    }
end

function SSHAndroid:onDispatcherRegisterActions()
    Dispatcher:registerAction("toggle_ssh_android_server", {
        category = "none",
        event = "ToggleSSHAndroidServer",
        title = _("Toggle SSH server (Android)"),
        general = true,
    })
end

return SSHAndroid
