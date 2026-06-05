local lfs = require("lfs")

local CONFIG = {
    request_path = lfs.writedir() .. "Logs/cv_capture_request.json",
    snapshot_dir = lfs.writedir() .. "Logs/cv_snapshots/",
    ack_path = lfs.writedir() .. "Logs/cv_screenshot_ack.json",
    pause_ack_path = lfs.writedir() .. "Logs/cv_pause_ack.json",
    screenshots_dir = lfs.writedir() .. "ScreenShots/",
    bbox_cache_path = lfs.writedir() .. "Logs/cv_bbox_cache.txt",
    object_pose_cache_path = lfs.writedir() .. "Logs/cv_object_pose_cache.txt",
    log_path = lfs.writedir() .. "Logs/cv_hook.log",
}

local last_processed_token = nil
local active = false
local bbox_refresh_interval_s = 3.0
local last_bbox_refresh_time = -1000000000
local last_object_pose_refresh_time = -1000000000
local last_pause_token = nil
local paused_by_cv = false

local function ensure_dir(path)
    os.execute('mkdir "' .. path .. '" >nul 2>nul')
end

local function append_log(message)
    local handle = io.open(CONFIG.log_path, "a")
    if not handle then
        return
    end
    handle:write(os.date("%Y-%m-%d %H:%M:%S") .. " | " .. tostring(message) .. "\n")
    handle:close()
end

local function read_all(path)
    local handle = io.open(path, "r")
    if not handle then
        return nil
    end
    local body = handle:read("*a")
    handle:close()
    return body
end

local function write_all(path, body)
    local handle = io.open(path, "w")
    if not handle then
        return false
    end
    handle:write(body)
    handle:close()
    return true
end

local function write_atomic(path, body)
    local temp_path = path .. ".tmp"

    if not write_all(temp_path, body) then
        return false
    end

    os.remove(path)

    local ok, err = os.rename(temp_path, path)
    if not ok then
        append_log("atomic write failed for " .. tostring(path) .. ": " .. tostring(err))
        return false
    end

    return true
end

local function json_encode(value)
    local value_type = type(value)
    if value == nil then
        return "null"
    end
    if value_type == "number" then
        return tostring(value)
    end
    if value_type == "boolean" then
        return value and "true" or "false"
    end
    if value_type == "string" then
        local escaped = value:gsub('\\', '\\\\'):gsub('"', '\\"')
        escaped = escaped:gsub('\n', '\\n')
        return '"' .. escaped .. '"'
    end
    if value_type == "table" then
        local is_array = true
        local count = 0
        for key, _ in pairs(value) do
            count = count + 1
            if type(key) ~= "number" then
                is_array = false
            end
        end
        local parts = {}
        if is_array then
            for index = 1, count do
                parts[#parts + 1] = json_encode(value[index])
            end
            return "[" .. table.concat(parts, ",") .. "]"
        end
        for key, nested in pairs(value) do
            parts[#parts + 1] = '"' .. tostring(key) .. '":' .. json_encode(nested)
        end
        return "{" .. table.concat(parts, ",") .. "}"
    end
    return '"unsupported"'
end

local function parse_request(body)
    if not body then
        return nil
    end
    return {
        status = body:match('"status"%s*:%s*"([^"]+)"'),
        frame_id = body:match('"frame_id"%s*:%s*"([^"]+)"'),
        frame_token = body:match('"frame_token"%s*:%s*"([^"]+)"'),
        capture_screenshot = body:match('"capture_screenshot"%s*:%s*(true)') ~= nil,
    }
end

local function snapshot_exists(frame_token)
    local path = CONFIG.snapshot_dir .. "snapshot_" .. frame_token .. ".json"
    local handle = io.open(path, "r")
    if not handle then
        return false
    end
    handle:close()
    return true
end

local function write_ack(frame_id, frame_token, status, screenshot_name, backend, error_message)
    local payload = {
        frame_id = frame_id,
        frame_token = frame_token,
        status = status,
        screenshot_name = screenshot_name,
        backend = backend,
        error = error_message,
    }
    local temp_path = CONFIG.ack_path .. ".tmp"
    write_all(temp_path, json_encode(payload))
    os.remove(CONFIG.ack_path)
    os.rename(temp_path, CONFIG.ack_path)
end

local function write_pause_ack(frame_id, frame_token, status, paused, stage, backend, error_message)
    local payload = {
        frame_id = frame_id,
        frame_token = frame_token,
        status = status,
        paused = paused,
        stage = stage,
        backend = backend,
        error = error_message,
        real_time_s = DCS.getRealTime(),
    }
    local temp_path = CONFIG.pause_ack_path .. ".tmp"
    write_all(temp_path, json_encode(payload))
    os.remove(CONFIG.pause_ack_path)
    os.rename(temp_path, CONFIG.pause_ack_path)
end

local function set_dcs_pause(paused)
    if not DCS or not DCS.setPause then
        return false, "DCS.setPause unavailable"
    end

    local ok, err = pcall(function()
        DCS.setPause(paused)
    end)

    if not ok then
        return false, tostring(err)
    end

    paused_by_cv = paused
    return true, nil
end

local function lua_quote(value)
    return string.format("%q", tostring(value or ""))
end

local function request_export_snapshot(request)
    if snapshot_exists(request.frame_token) then
        return true
    end

    if not net or not net.dostring_in then
        append_log("Export snapshot request failed: net.dostring_in unavailable")
        return false
    end

    local script = string.format(
        "return CV_DATASET_CAPTURE_NOW(%s,%s,%s)",
        lua_quote(request.frame_id),
        lua_quote(request.frame_token),
        request.capture_screenshot and "true" or "false"
    )

    local ok, result = pcall(net.dostring_in, "export", script)
    if not ok then
        append_log("Export snapshot request pcall failed for token=" .. tostring(request.frame_token) .. ": " .. tostring(result))
        return false
    end

    if not snapshot_exists(request.frame_token) then
        append_log("Export snapshot request did not create snapshot for token=" .. tostring(request.frame_token) .. ": " .. tostring(result))
        return false
    end

    append_log("Export snapshot requested for token=" .. tostring(request.frame_token) .. ": " .. tostring(result))
    return true
end

local function pause_for_request(request, stage)
    if paused_by_cv and last_pause_token == request.frame_token then
        return true
    end

    local ok, err = set_dcs_pause(true)
    if ok then
        write_pause_ack(request.frame_id, request.frame_token, "paused", true, stage, "DCS.setPause", nil)
        append_log("DCS paused for token=" .. tostring(request.frame_token) .. " stage=" .. tostring(stage))
        last_pause_token = request.frame_token
        return true
    end

    write_pause_ack(request.frame_id or "unknown", request.frame_token or "unknown", "error", false, stage, "DCS.setPause", err)
    append_log("DCS pause failed for token=" .. tostring(request.frame_token) .. ": " .. tostring(err))
    last_pause_token = request.frame_token
    return false
end

local function release_pause_after_capture(request)
    if not paused_by_cv then
        return
    end

    local ok, err = set_dcs_pause(false)
    if ok then
        append_log("DCS unpaused after capture token=" .. tostring(request.frame_token))
    else
        append_log("DCS unpause failed for token=" .. tostring(request.frame_token) .. ": " .. tostring(err))
    end
end

local function refresh_bbox_cache(force)
    if not active then
        return
    end
    local now = DCS.getRealTime()
    if not force and (now - last_bbox_refresh_time) < bbox_refresh_interval_s then
        return
    end

    last_bbox_refresh_time = now
    local script = [=[
return (function()
    local function sanitize(value)
        if value == nil then
            return ""
        end

        local text = tostring(value)
        text = text:gsub("\t", " ")
        text = text:gsub("\n", " ")
        text = text:gsub("\r", " ")
        return text
    end

    local lines = {}
    local sides = {
        coalition.side.RED,
        coalition.side.BLUE,
        coalition.side.NEUTRAL
    }

    local categories = {
        Group.Category.AIRPLANE,
        Group.Category.HELICOPTER,
        Group.Category.GROUND,
        Group.Category.SHIP
    }

    for _, side in ipairs(sides) do
        for _, category in ipairs(categories) do
            local groups = coalition.getGroups(side, category)

            if groups then
                for _, group in ipairs(groups) do
                    local units = group:getUnits()

                    if units then
                        for _, unit in ipairs(units) do
                            local ok = pcall(function()
                                local desc = unit:getDesc()

                                if desc and desc.box then
                                    local box = desc.box
                                    local bmin = box.min
                                    local bmax = box.max

                                    if bmin and bmax
                                        and bmin.x and bmin.y and bmin.z
                                        and bmax.x and bmax.y and bmax.z
                                    then
                                        lines[#lines + 1] = string.format(
                                            "%s\t%s\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f",
                                            sanitize(unit:getName()),
                                            sanitize(desc.typeName or desc.displayName or unit:getTypeName() or "unknown"),
                                            bmin.x,
                                            bmin.y,
                                            bmin.z,
                                            bmax.x,
                                            bmax.y,
                                            bmax.z
                                        )
                                    end
                                end
                            end)
                        end
                    end
                end
            end
        end
    end

    return table.concat(lines, "\n")
end)()
]=]
    local ok, result = false, nil

    if a_do_script then
        ok, result = pcall(a_do_script, script)
        if not ok then
            append_log("BBox cache refresh failed via a_do_script")
        end
    end

    if (not ok) and net and net.dostring_in then
        ok, result = pcall(net.dostring_in, 'scripting', script)
    end

    if not ok then
        append_log("BBox cache refresh failed: pcall error")
        return
    end

    local payload = ""
    if type(result) == "string" then
        payload = result
    elseif result ~= nil then
        payload = tostring(result)
    end

    if payload ~= nil and payload ~= "" then
        append_log("BBox cache refresh success, bytes=" .. tostring(string.len(payload)))
        write_atomic(CONFIG.bbox_cache_path, payload)
    else
        append_log("BBox cache refresh empty payload")
    end
end

local function refresh_object_pose_cache(force)
    if not active then
        return
    end
    local now = DCS.getRealTime()
    if not force and (now - last_object_pose_refresh_time) < bbox_refresh_interval_s then
        return
    end

    last_object_pose_refresh_time = now
    local script = [=[
return (function()
    local function sanitize(value)
        if value == nil then
            return ""
        end

        local text = tostring(value)
        text = text:gsub("\t", " ")
        text = text:gsub("\n", " ")
        text = text:gsub("\r", " ")
        return text
    end

    local function valid_vec(vec)
        return vec and vec.x and vec.y and vec.z
    end

    local function log_pose_cache_issue(message)
        pcall(function()
            if log and log.write then
                log.write("cv_dataset_hook", log.WARNING or log.INFO or 2, message)
            elseif env and env.info then
                env.info("cv_dataset_hook: " .. tostring(message))
            end
        end)
    end

    local lines = {}
    local sides = {
        coalition.side.RED,
        coalition.side.BLUE,
        coalition.side.NEUTRAL
    }

    local categories = {
        Group.Category.AIRPLANE,
        Group.Category.HELICOPTER,
        Group.Category.GROUND,
        Group.Category.SHIP
    }

    for _, side in ipairs(sides) do
        for _, category in ipairs(categories) do
            local groups = coalition.getGroups(side, category)

            if groups then
                for _, group in ipairs(groups) do
                    local units = group:getUnits()

                    if units then
                        for _, unit in ipairs(units) do
                            local ok, err = pcall(function()
                                local unit_name = sanitize(unit:getName())
                                local pos = unit:getPosition()

                                if pos and valid_vec(pos.p) and valid_vec(pos.x) and valid_vec(pos.y) and valid_vec(pos.z) then
                                    lines[#lines + 1] = string.format(
                                        "%s\t%s\t%s\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f",
                                        sanitize(unit:getID()),
                                        sanitize(unit:getName()),
                                        sanitize(unit:getTypeName() or "unknown"),
                                        pos.p.x,
                                        pos.p.y,
                                        pos.p.z,
                                        pos.x.x,
                                        pos.x.y,
                                        pos.x.z,
                                        pos.y.x,
                                        pos.y.y,
                                        pos.y.z,
                                        pos.z.x,
                                        pos.z.y,
                                        pos.z.z
                                    )
                                else
                                    log_pose_cache_issue("object pose cache skipped unit with unavailable getPosition: " .. tostring(unit_name))
                                end
                            end)
                            if not ok then
                                log_pose_cache_issue("object pose cache unit:getPosition failed: " .. tostring(err))
                            end
                        end
                    end
                end
            end
        end
    end

    return table.concat(lines, "\n")
end)()
]=]
    local ok, result = false, nil

    if a_do_script then
        ok, result = pcall(a_do_script, script)
        if not ok then
            append_log("Object pose cache refresh failed via a_do_script")
        end
    end

    if (not ok) and net and net.dostring_in then
        ok, result = pcall(net.dostring_in, 'scripting', script)
    end

    if not ok then
        append_log("Object pose cache refresh failed: pcall error")
        return
    end

    local payload = ""
    if type(result) == "string" then
        payload = result
    elseif result ~= nil then
        payload = tostring(result)
    end

    if payload ~= nil and payload ~= "" then
        append_log("Object pose cache refresh success, bytes=" .. tostring(string.len(payload)))
        write_atomic(CONFIG.object_pose_cache_path, payload)
    else
        append_log("Object pose cache refresh empty payload")
    end
end

local function perform_capture(request)
    if not pause_for_request(request, "capture_pending") then
        return
    end

    refresh_bbox_cache(true)
    refresh_object_pose_cache(true)

    if not request_export_snapshot(request) then
        return
    end

    if not request.capture_screenshot then
        write_ack(
            request.frame_id,
            request.frame_token,
            "done",
            nil,
            "no_screenshot",
            nil
        )
        last_processed_token = request.frame_token
        release_pause_after_capture(request)
        return
    end

    local screenshot_name = request.frame_id .. "_" .. request.frame_token .. ".png"
    os.remove(CONFIG.screenshots_dir .. screenshot_name)

    local ok, err = pcall(function()
        DCS.makeScreenShot(screenshot_name)
    end)

    if ok then
        write_ack(request.frame_id, request.frame_token, "done", screenshot_name, "DCS.makeScreenShot", nil)
        append_log("Screenshot captured for token=" .. tostring(request.frame_token))
    else
        write_ack(request.frame_id, request.frame_token, "error", screenshot_name, "DCS.makeScreenShot", tostring(err))
        append_log("Screenshot failed for token=" .. tostring(request.frame_token) .. ": " .. tostring(err))
    end

    last_processed_token = request.frame_token
    release_pause_after_capture(request)
end

local hook_callbacks = {}

function hook_callbacks.onSimulationStart()
    ensure_dir(CONFIG.snapshot_dir)
    active = true
    local ok, err = set_dcs_pause(true)
    if ok then
        write_pause_ack("startup", "startup", "paused", true, "simulation_start", "DCS.setPause", nil)
        append_log("cv hook started and simulation paused")
    else
        write_pause_ack("startup", "startup", "error", false, "simulation_start", "DCS.setPause", err)
        append_log("cv hook started but pause failed: " .. tostring(err))
    end
end

function hook_callbacks.onSimulationStop()
    active = false
    append_log("cv hook stopped")
end

function hook_callbacks.onSimulationFrame()
    if not active then
        return
    end
    refresh_bbox_cache(false)
    refresh_object_pose_cache(false)
    local request = parse_request(read_all(CONFIG.request_path))
    if not request then
        return
    end
    if request.status ~= "pending" then
        if request.status == "pause_pending" then
            pause_for_request(request, "pause_pending")
        end
        return
    end
    if not request.frame_token or request.frame_token == last_processed_token then
        return
    end
    if not request.frame_id then
        write_ack("unknown", request.frame_token or "unknown", "error", nil, "hook_parse", "missing frame_id")
        return
    end
    perform_capture(request)
end

DCS.setUserCallbacks(hook_callbacks)
