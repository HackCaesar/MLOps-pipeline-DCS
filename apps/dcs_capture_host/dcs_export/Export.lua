local lfs = require("lfs")

local CONFIG = {
    request_path = lfs.writedir() .. "Logs/cv_capture_request.json",
    snapshot_dir = lfs.writedir() .. "Logs/cv_snapshots/",
    bbox_cache_path = lfs.writedir() .. "Logs/cv_bbox_cache.txt",
    object_pose_cache_path = lfs.writedir() .. "Logs/cv_object_pose_cache.txt",
    camera_request_path = lfs.writedir() .. "Logs/cv_camera_request.json",
    camera_ack_path = lfs.writedir() .. "Logs/cv_camera_ack.json",
    log_path = lfs.writedir() .. "Logs/cv_export.log",
}

local last_capture_token = nil
local last_camera_token = nil
local pending_camera_request = nil
local bbox_cache_by_unit = {}
local bbox_cache_by_type = {}
local object_pose_cache_by_id = {}
local object_pose_cache_by_unit = {}

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
            parts[#parts + 1] = json_encode(tostring(key)) .. ":" .. json_encode(nested)
        end
        return "{" .. table.concat(parts, ",") .. "}"
    end
    return '"unsupported"'
end

local function parse_capture_request(body)
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

local function table_count(tbl)
    local count = 0
    for _, _ in pairs(tbl) do
        count = count + 1
    end
    return count
end

local function parse_vector(body, vector_name)
    if not body or not vector_name then
        return nil
    end 
    local number_pattern = "([%+%-%deE%.]+)"

    local x = body:match('"' .. vector_name .. '"%s*:%s*%{[^}]-"x"%s*:%s*' .. number_pattern)
    local y = body:match('"' .. vector_name .. '"%s*:%s*%{[^}]-"y"%s*:%s*' .. number_pattern)
    local z = body:match('"' .. vector_name .. '"%s*:%s*%{[^}]-"z"%s*:%s*' .. number_pattern)
    if not x or not y or not z then
        return nil
    end

    local nx = tonumber(x)
    local ny = tonumber(y)    
    local nz = tonumber(z)

    if not nx or not ny or not nz then
        return nil
    end

    return {
        x = nx, y = ny, z = nz
    }
end

local function load_bbox_cache()
    bbox_cache_by_unit = {}
    bbox_cache_by_type = {}

    local body = read_all(CONFIG.bbox_cache_path)
    if not body or body == "" then
        append_log("bbox cache is empty or not found: " .. tostring(CONFIG.bbox_cache_path))
        return
    end

    for line in string.gmatch(body, "[^\r\n]+") do
        local unit_name, type_name, min_x, min_y, min_z, max_x, max_y, max_z =
            line:match("^([^\t]*)\t([^\t]*)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)$")
        if not unit_name or not type_name then
            append_log("bad bbox cache line format: " .. tostring(line))
        else
            local n_min_x = tonumber(min_x)
            local n_min_y = tonumber(min_y)
            local n_min_z = tonumber(min_z)

            local n_max_x = tonumber(max_x)
            local n_max_y = tonumber(max_y)
            local n_max_z = tonumber(max_z)

            if not n_min_x or not n_min_y or not n_min_z or not n_max_x or not n_max_y or not n_max_z then
                append_log(
                    "bad bbox cache numeric values: "
                    .. "unit=" .. tostring(unit_name)
                    .. ", type=" .. tostring(type_name)  
                    .. ", min=(" .. tostring(min_x) .. "," .. tostring(min_y) .. "," .. tostring(min_z) .. ")" 
                    .. ", max=(" .. tostring(max_x) .. "," .. tostring(max_y) .. "," .. tostring(max_z) .. ")"         
                )
            else
                local entry = {
                    bbox_min = {
                        x = n_min_x,
                        y = n_min_y,
                        z = n_min_z,
                    },
                    bbox_max = {
                        x = n_max_x,
                        y = n_max_y,
                        z = n_max_z,
                    },
                }
            
                if unit_name ~= "" then
                    bbox_cache_by_unit[unit_name] = entry
                end
                if type_name ~= "" and bbox_cache_by_type[type_name] == nil then
                    bbox_cache_by_type[type_name] = entry
                end
            end
        end
    end

    append_log(
        "bbox cache loaded: "
        .. "units=" .. tostring(table_count(bbox_cache_by_unit))
        .. ", types=" .. tostring(table_count(bbox_cache_by_type))        
    )
end

local function load_object_pose_cache()
    object_pose_cache_by_id = {}
    object_pose_cache_by_unit = {}

    local body = read_all(CONFIG.object_pose_cache_path)
    if not body or body == "" then
        append_log("object pose cache is empty or not found: " .. tostring(CONFIG.object_pose_cache_path))
        return
    end

    for line in string.gmatch(body, "[^\r\n]+") do
        local unit_id, unit_name, type_name,
            p_x, p_y, p_z,
            x_x, x_y, x_z,
            y_x, y_y, y_z,
            z_x, z_y, z_z =
            line:match(
                "^([^\t]*)\t([^\t]*)\t([^\t]*)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)\t([^\t]+)$"
            )

        if not unit_id or not unit_name or not type_name then
            append_log("bad object pose cache line format: " .. tostring(line))
        else
            local values = {
                tonumber(p_x), tonumber(p_y), tonumber(p_z),
                tonumber(x_x), tonumber(x_y), tonumber(x_z),
                tonumber(y_x), tonumber(y_y), tonumber(y_z),
                tonumber(z_x), tonumber(z_y), tonumber(z_z),
            }
            local valid = true
            for _, value in ipairs(values) do
                if value == nil then
                    valid = false
                    break
                end
            end

            if not valid then
                append_log("bad object pose cache numeric values: unit_id=" .. tostring(unit_id) .. ", unit=" .. tostring(unit_name))
            else
                local entry = {
                    unit_id = unit_id,
                    unit_name = unit_name,
                    type_name = type_name,
                    p = {x = values[1], y = values[2], z = values[3]},
                    x = {x = values[4], y = values[5], z = values[6]},
                    y = {x = values[7], y = values[8], z = values[9]},
                    z = {x = values[10], y = values[11], z = values[12]},
                }

                if unit_id ~= "" then
                    object_pose_cache_by_id[unit_id] = entry
                end
                if unit_name ~= "" and object_pose_cache_by_unit[unit_name] == nil then
                    object_pose_cache_by_unit[unit_name] = entry
                end
            end
        end
    end

    append_log(
        "object pose cache loaded: "
        .. "ids=" .. tostring(table_count(object_pose_cache_by_id))
        .. ", units=" .. tostring(table_count(object_pose_cache_by_unit))
    )
end

local function parse_camera_request(body)
    if not body then
        return nil
    end
    return {
        status = body:match('"status"%s*:%s*"([^"]+)"'),
        request_token = body:match('"request_token"%s*:%s*"([^"]+)"'),
        
        position = parse_vector(body, "position"),

        axis_x = parse_vector(body, "axis_x"),
        axis_y = parse_vector(body, "axis_y"),
        axis_z = parse_vector(body, "axis_z"),
    }
end

local function camera_request_to_dcs_basis(request)
    return {
        p = {
            x = request.position.x,
            y = request.position.y,
            z = request.position.z,
        },

        x = {
            x = request.axis_x.x,
            y = request.axis_x.y,
            z = request.axis_x.z,
        },

        y = {
            x = request.axis_y.x,
            y = request.axis_y.y,
            z = request.axis_y.z,
        },

        z = {
            x = request.axis_z.x,
            y = request.axis_z.y,
            z = request.axis_z.z,
        },
    }
end

local function basis_to_table(basis)
    if not basis then
        return nil
    end

    if not basis.x or not basis.y or not basis.z then
        return nil
    end

    return {
        axis_x = {
            x = basis.x.x,
            y = basis.x.y,
            z = basis.x.z,
        },
        axis_y = {
            x = basis.y.x,
            y = basis.y.y,
            z = basis.y.z,
        },
        axis_z = {
            x = basis.z.x,
            y = basis.z.y,
            z = basis.z.z,
        },
    }
end

local function collect_camera_state()
    local camera = LoGetCameraPosition()
    if not camera then
        return nil
    end
    return {
        world_position_m = {
            x = camera.p.x,
            y = camera.p.y,
            z = camera.p.z,
        },
        world_basis_raw = basis_to_table(camera),
    }
end

local function collect_object_states()
    local objects = LoGetWorldObjects()
    local result = {}
    if not objects then
        return result
    end

    load_bbox_cache()
    load_object_pose_cache()

    for id, object_data in pairs(objects) do
        local world_position_m = nil
        local lat_long_alt = nil
        local world_basis_raw = nil
        local position_source = "missing"
        local object_pose_cache_match = nil

        local unit_name = object_data.UnitName or object_data.unitName or ""
        local bbox_entry = nil
        local pose_entry = object_pose_cache_by_id[tostring(id)]

        if pose_entry then
            object_pose_cache_match = "id"
        elseif unit_name ~= "" then
            pose_entry = object_pose_cache_by_unit[unit_name]
            if pose_entry then
                object_pose_cache_match = "unit_name"
            end
        end

        if object_data.Position then
            if object_data.Position.p then
                world_position_m = {
                    x = object_data.Position.p.x,
                    y = object_data.Position.p.y,
                    z = object_data.Position.p.z,
                }
                position_source = "Position.p"

                if object_data.Position.x and type(object_data.Position.x) == "table" then
                    world_basis_raw = basis_to_table(object_data.Position)
                end

            elseif type(object_data.Position.x) == "number" then
                world_position_m = {
                    x = object_data.Position.x,
                    y = object_data.Position.y,
                    z = object_data.Position.z,
                }
                position_source = "Position.xyz"
            end
        end

        if pose_entry then
            world_position_m = {
                x = pose_entry.p.x,
                y = pose_entry.p.y,
                z = pose_entry.p.z,
            }
            world_basis_raw = basis_to_table(pose_entry)
            position_source = "object_pose_cache"
        end

        if object_data.LatLongAlt then
            lat_long_alt = {
                lat = object_data.LatLongAlt.Lat,
                long = object_data.LatLongAlt.Long,
                alt = object_data.LatLongAlt.Alt,
            }
        end

        if unit_name ~= "" then
            bbox_entry = bbox_cache_by_unit[unit_name]
        end

        if bbox_entry == nil then
            local possible_type_names = {
                object_data.Name,
                object_data.TypeName,
                object_data.typeName,
                object_data.DisplayName,
                object_data.displayName,
            }

            for _, possible_type_name in ipairs(possible_type_names) do
                if possible_type_name and bbox_cache_by_type[possible_type_name] then
                    bbox_entry = bbox_cache_by_type[possible_type_name]
                    break
                end
            end
        end

        result[#result + 1] = {
            id = tostring(id),
            
            type_name = object_data.Name or "unknown",
            unit_name = unit_name,
            type_levels = object_data.Type,

            world_position_m = world_position_m,
            lat_long_alt = lat_long_alt,
            position_source = position_source,
            object_pose_cache_match = object_pose_cache_match,

            world_basis_raw = world_basis_raw,

            heading_rad = object_data.Heading or object_data.heading or 0,
            pitch_rad = object_data.Pitch or object_data.pitch or 0,
            bank_rad = object_data.Bank or object_data.bank or 0,

            bbox_min_local = bbox_entry and bbox_entry.bbox_min or nil,
            bbox_max_local = bbox_entry and bbox_entry.bbox_max or nil,
                    
            has_world_position = world_position_m ~= nil,
            has_world_basis = world_basis_raw ~= nil,
            has_bbox = bbox_entry ~= nil,
                    
            geometry_source = bbox_entry and "bbox_cache" or "missing_bbox",
        }
    end
    return result
end

local function build_snapshot(request)
    return {
        schema_version = "1.0",
        frame_id = request.frame_id,
        frame_token = request.frame_token,
        capture_screenshot = request.capture_screenshot,
        simulation_time_s = LoGetModelTime(),
        camera = collect_camera_state(),
        objects = collect_object_states(),
    }
end

local function write_snapshot(request)
    local snapshot = build_snapshot(request)
    local output_path = CONFIG.snapshot_dir .. "snapshot_" .. request.frame_token .. ".json"
    local temp_path = output_path .. ".tmp"
    write_all(temp_path, json_encode(snapshot))
    os.remove(output_path)
    os.rename(temp_path, output_path)
    append_log("Snapshot written for token=" .. tostring(request.frame_token))
end

function CV_DATASET_CAPTURE_NOW(frame_id, frame_token, capture_screenshot)
    if not frame_token or frame_token == "" then
        append_log("CV_DATASET_CAPTURE_NOW failed: missing frame_token")
        return "error:missing_frame_token"
    end

    local request = {
        frame_id = frame_id or "unknown",
        frame_token = frame_token,
        capture_screenshot = capture_screenshot == true,
    }

    write_snapshot(request)
    last_capture_token = frame_token
    return "snapshot_written:" .. tostring(frame_token)
end

local function write_camera_ack(request_token, status, error_message, camera_state)
    local payload = {
        request_token = request_token,
        status = status,
        error = error_message,
        simulation_time_s = LoGetModelTime(),
        camera = camera_state,
    }
    local temp_path = CONFIG.camera_ack_path .. ".tmp"
    write_all(temp_path, json_encode(payload))
    os.remove(CONFIG.camera_ack_path)
    os.rename(temp_path, CONFIG.camera_ack_path)
end

local function apply_camera_request()
    local request = parse_camera_request(read_all(CONFIG.camera_request_path))
    if not request then
        pending_camera_request = nil
        return
    end
    if request.status ~= "pending" then
        pending_camera_request = nil
        return
    end
    if not request.request_token or request.request_token == last_camera_token then
        return
    end
    if not request.position or not request.axis_x or not request.axis_y or not request.axis_z then
        write_camera_ack(
            request.request_token or "unknown", 
            "error", 
            "camera request missing vectors", 
            collect_camera_state()
        )
        return
    end

    if pending_camera_request == nil or pending_camera_request.request_token ~= request.request_token then
        pending_camera_request = {
            request_token = request.request_token,
            position = request.position,
            axis_x = request.axis_x,
            axis_y = request.axis_y,
            axis_z = request.axis_z,
            stage = "jump",
        }
        pcall(function()
            LoSetCommand(158)
        end)
        append_log("Camera request entered free-camera stage token=" .. tostring(request.request_token))
        return
    end

    if pending_camera_request.stage == "jump" then
        local current_camera = LoGetCameraPosition()
        if not current_camera or not LoSetCameraPosition then
            write_camera_ack(request.request_token, "error", "LoSetCameraPosition unavailable", collect_camera_state())
            last_camera_token = request.request_token
            pending_camera_request = nil
            return
        end

        local updated_camera = camera_request_to_dcs_basis(request)

        local ok, err = pcall(function()
            LoSetCameraPosition(updated_camera)
        end)

        if ok then
            pending_camera_request.stage = "verify"
            append_log("Camera request set-position stage token=" .. tostring(request.request_token))
        else
            write_camera_ack(request.request_token, "error", tostring(err), collect_camera_state())
            append_log("Camera request failed token=" .. tostring(request.request_token) .. ": " .. tostring(err))
            last_camera_token = request.request_token
            pending_camera_request = nil
        end
        return
    end

    if pending_camera_request.stage == "verify" then
        write_camera_ack(request.request_token, "applied", nil, collect_camera_state())
        append_log("Camera request applied token=" .. tostring(request.request_token))
        last_camera_token = request.request_token
        pending_camera_request = nil
    end
end

local function process_capture_request()
    local request = parse_capture_request(read_all(CONFIG.request_path))
    if not request then
        return
    end
    if request.status ~= "pending" then
        return
    end
    if not request.frame_token or request.frame_token == last_capture_token then
        return
    end
    if not request.frame_id then
        append_log("Capture request missing frame_id")
        return
    end
    write_snapshot(request)
    last_capture_token = request.frame_token
end

function LuaExportStart()
    ensure_dir(CONFIG.snapshot_dir)
    append_log("cv exporter started")
end

function LuaExportBeforeNextFrame()
    apply_camera_request()
end

function LuaExportAfterNextFrame()
end

function LuaExportStop()
    append_log("cv exporter stopped")
end
