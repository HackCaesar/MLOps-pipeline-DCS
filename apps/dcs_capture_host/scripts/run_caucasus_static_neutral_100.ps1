<#
.SYNOPSIS
    Generate 10 missions, capture 10 frames each, validate per-mission, then
    rebuild the merged COCO file from per-frame metadata.

.DESCRIPTION
    Drives the existing pipeline. Does NOT change orchestrator/main.py,
    exporters or projection math. It only:

      1) regenerates the 10 .miz files via scenario_builder;
      2) loops through them in canonical order, calling orchestrator/main.py
         with --frames 10 and --frame-id <mission stem>;
      3) validates each captured batch with validate_mission_capture.py;
         if a mission fails, *stops* unless -ContinueOnMissionFail is set;
      4) after the loop, rebuilds dataset/annotations/_annotations.coco.json
         from dataset/metadata/*.json so the final COCO contains all 100
         frames (the per-process exporter would otherwise overwrite previous
         batches every time orchestrator/main.py is restarted);
      5) runs the dataset-wide validator and writes the final batch report.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass `
        -File D:\yolo\DCS_AutoDataset\scripts\run_caucasus_static_neutral_100.ps1
#>

param(
    [string]$ProjectRoot         = "D:\yolo\DCS_AutoDataset",
    [string]$PythonExe           = "D:\yolo\.dcs_venv\Scripts\python.exe",
    [string]$Config              = "D:\yolo\DCS_AutoDataset\configs\caucasus_static_neutral_capture.yaml",
    [int]   $FramesPerMission    = 10,
    [int]   $MinUsableFrames     = 8,
    [int]   $MinUniquePositions  = 8,
    [switch]$SkipMissionGen,
    [switch]$ContinueOnMissionFail
)

$ErrorActionPreference = "Stop"
$startTime = Get-Date

$missionDir     = Join-Path $ProjectRoot "missions\caucasus_static_neutral_100"
$datasetRoot    = Join-Path $ProjectRoot "dataset"
$diagnosticsDir = Join-Path $ProjectRoot "diagnostics"
$cocoOutput     = Join-Path $datasetRoot "annotations\_annotations.coco.json"
$batchReport    = Join-Path $diagnosticsDir "static_neutral_batch_report.json"

# Canonical mission list — keep in sync with MISSIONS in scenario_builder.
$missions = @(
    @{ Index = 1;  Prefix = "caucasus_static_neutral_01_winter_morning_clear_ships";      File = "caucasus_static_neutral_01_winter_morning_clear_ships.miz" },
    @{ Index = 2;  Prefix = "caucasus_static_neutral_02_winter_day_cloudy_ships";         File = "caucasus_static_neutral_02_winter_day_cloudy_ships.miz" },
    @{ Index = 3;  Prefix = "caucasus_static_neutral_03_spring_morning_haze_helos";       File = "caucasus_static_neutral_03_spring_morning_haze_helos.miz" },
    @{ Index = 4;  Prefix = "caucasus_static_neutral_04_spring_evening_broken_planes";    File = "caucasus_static_neutral_04_spring_evening_broken_planes.miz" },
    @{ Index = 5;  Prefix = "caucasus_static_neutral_05_summer_day_clear_ships";          File = "caucasus_static_neutral_05_summer_day_clear_ships.miz" },
    @{ Index = 6;  Prefix = "caucasus_static_neutral_06_summer_evening_warm_helos";       File = "caucasus_static_neutral_06_summer_evening_warm_helos.miz" },
    @{ Index = 7;  Prefix = "caucasus_static_neutral_07_autumn_morning_overcast_planes";  File = "caucasus_static_neutral_07_autumn_morning_overcast_planes.miz" },
    @{ Index = 8;  Prefix = "caucasus_static_neutral_08_autumn_day_cloudy_ships";         File = "caucasus_static_neutral_08_autumn_day_cloudy_ships.miz" },
    @{ Index = 9;  Prefix = "caucasus_static_neutral_09_spring_day_mist_helos";           File = "caucasus_static_neutral_09_spring_day_mist_helos.miz" },
    @{ Index = 10; Prefix = "caucasus_static_neutral_10_autumn_evening_complex_planes";   File = "caucasus_static_neutral_10_autumn_evening_complex_planes.miz" }
)

# Mission generation (skippable when iterating on the runner itself).
if (-not $SkipMissionGen) {
    Write-Host "===== Generating 10 mission files =====" -ForegroundColor Cyan
    & $PythonExe (Join-Path $ProjectRoot "scenario_builder\create_caucasus_static_neutral_missions.py") `
        --output-dir $missionDir
    if ($LASTEXITCODE -ne 0) {
        throw "scenario_builder failed with exit code $LASTEXITCODE"
    }
}

# Loop through missions.
$results = @()
foreach ($mission in $missions) {
    $missionPath = Join-Path $missionDir $mission.File
    Write-Host ""
    Write-Host ("===== [{0}/10] {1} =====" -f $mission.Index, $mission.Prefix) -ForegroundColor Cyan
    Write-Host "Mission file: $missionPath"

    $captureExit = 0
    try {
        & $PythonExe (Join-Path $ProjectRoot "orchestrator\main.py") `
            --config $Config `
            --mission $missionPath `
            --frames $FramesPerMission `
            --frame-id $mission.Prefix
        $captureExit = $LASTEXITCODE
    } catch {
        Write-Host "Capture raised: $_" -ForegroundColor Red
        $captureExit = 1
    }

    if ($captureExit -ne 0) {
        $entry = @{
            mission   = $mission.Prefix
            captured  = $false
            validated = $false
            error     = "orchestrator exit=$captureExit"
        }
        $results += $entry
        if (-not $ContinueOnMissionFail) {
            $results | ConvertTo-Json -Depth 5 | Set-Content -Path $batchReport -Encoding UTF8
            throw ("orchestrator/main.py failed on {0} (exit={1}); stopping batch." -f `
                $mission.Prefix, $captureExit)
        }
        continue
    }

    Write-Host "----- per-mission validation -----" -ForegroundColor Yellow
    $reportPath = Join-Path $diagnosticsDir ("mission_{0}_report.json" -f $mission.Prefix)
    & $PythonExe (Join-Path $ProjectRoot "validators\validate_mission_capture.py") `
        --dataset-root $datasetRoot `
        --mission-prefix $mission.Prefix `
        --expected-frames $FramesPerMission `
        --min-usable-frames $MinUsableFrames `
        --min-unique-camera-positions $MinUniquePositions `
        --report $reportPath
    $validateExit = $LASTEXITCODE

    $entry = @{
        mission       = $mission.Prefix
        captured      = $true
        validated     = ($validateExit -eq 0)
        report_path   = $reportPath
        validate_exit = $validateExit
    }
    $results += $entry

    if ($validateExit -ne 0 -and (-not $ContinueOnMissionFail)) {
        $results | ConvertTo-Json -Depth 5 | Set-Content -Path $batchReport -Encoding UTF8
        throw ("per-mission validator rejected {0}; see {1}. Stopping batch." -f `
            $mission.Prefix, $reportPath)
    }
}

# Rebuild merged COCO from per-frame metadata.
Write-Host ""
Write-Host "===== Rebuilding merged COCO from dataset/metadata =====" -ForegroundColor Cyan
& $PythonExe (Join-Path $ProjectRoot "scripts\rebuild_coco_from_metadata.py") `
    --metadata-dir (Join-Path $datasetRoot "metadata") `
    --output $cocoOutput
if ($LASTEXITCODE -ne 0) {
    throw "rebuild_coco_from_metadata.py failed with exit code $LASTEXITCODE"
}

# Dataset-wide validator.
Write-Host ""
Write-Host "===== Dataset-wide validation =====" -ForegroundColor Cyan
& $PythonExe (Join-Path $ProjectRoot "validators\validate_static_neutral_dataset.py") `
    --dataset-root $datasetRoot `
    --mission-prefix "caucasus_static_neutral_" `
    --report (Join-Path $diagnosticsDir "static_neutral_dataset_report.json")

# Final batch report.
$batch = @{
    started_at         = $startTime.ToString("o")
    finished_at        = (Get-Date).ToString("o")
    frames_per_mission = $FramesPerMission
    missions_total     = $missions.Count
    missions_captured  = ($results | Where-Object { $_.captured }).Count
    missions_validated = ($results | Where-Object { $_.validated }).Count
    results            = $results
}
$batch | ConvertTo-Json -Depth 5 | Set-Content -Path $batchReport -Encoding UTF8

Write-Host ""
Write-Host "===== BATCH COMPLETE =====" -ForegroundColor Green
Write-Host ("captured  : {0}/{1}" -f $batch.missions_captured,  $batch.missions_total)
Write-Host ("validated : {0}/{1}" -f $batch.missions_validated, $batch.missions_total)
Write-Host ("coco      : {0}" -f $cocoOutput)
Write-Host ("report    : {0}" -f $batchReport)
