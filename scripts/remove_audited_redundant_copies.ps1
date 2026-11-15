$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$auditPath = Join-Path $workspaceRoot 'outputs\storage_cleanup\pre_deletion_audit.json'
$audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
if ($audit.status -ne 'PASS') { throw 'Pre-deletion content/dependency audit did not pass.' }
$expectedTargets = @('data/yelp_photos/yelp_photos', 'data/processed/.yelp_2300_v1.serial_attempt')
if (($audit.delete_targets -join '|') -ne ($expectedTargets -join '|')) { throw 'Unexpected deletion scope.' }
$resolvedTargets = @()
foreach ($relativeTarget in $expectedTargets) {
    $candidate = Join-Path $workspaceRoot $relativeTarget
    $resolved = (Resolve-Path -LiteralPath $candidate).Path
    if (-not $resolved.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $resolved" }
    $item = Get-Item -LiteralPath $resolved -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point refused: $resolved" }
    $resolvedTargets += $resolved
}
$archiveItem = Get-Item -LiteralPath $resolvedTargets[0]
if ($archiveItem.PSIsContainer -or $archiveItem.Length -ne $audit.archive.bytes) { throw 'Archive no longer matches audit.' }
$partialItems = @(Get-ChildItem -LiteralPath $resolvedTargets[1] -Recurse -Force)
if (@($partialItems | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Reparse point within partial attempt.' }
$partialFiles = @($partialItems | Where-Object { -not $_.PSIsContainer })
$partialBytes = ($partialFiles | Measure-Object -Property Length -Sum).Sum
if ($partialFiles.Count -ne $audit.serial_attempt.files -or $partialBytes -ne $audit.serial_attempt.bytes) { throw 'Partial attempt no longer matches audit.' }
foreach ($preserved in $audit.serial_attempt.different_small_provenance_preserved) {
    if (-not (Test-Path -LiteralPath (Join-Path $workspaceRoot $preserved) -PathType Leaf)) { throw 'Missing preserved provenance.' }
}
foreach ($required in @('data\yelp_photos\photos', 'data\yelp_photos\photos.json', 'data\processed\yelp_2300_v1', 'data\processed\yelp_2300_english_v1', 'data\processed\splits')) {
    if (-not (Test-Path -LiteralPath (Join-Path $workspaceRoot $required))) { throw "Missing current source: $required" }
}
$drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($workspaceRoot))
$freeBefore = $drive.AvailableFreeSpace
Remove-Item -LiteralPath $resolvedTargets[0] -Force
Remove-Item -LiteralPath $resolvedTargets[1] -Recurse -Force
foreach ($target in $resolvedTargets) {
    if (Test-Path -LiteralPath $target) { throw "Deletion incomplete: $target" }
}
$report = [ordered]@{
    status = 'PASS'
    authorization = 'User explicitly requests removing unnecessary project files/data.'
    deleted_absolute_paths = $resolvedTargets
    deleted_logical_bytes = [long]$audit.archive.bytes + [long]$audit.serial_attempt.bytes
    partial_provenance_preserved = $audit.serial_attempt.different_small_provenance_preserved
    disk_free_before = $freeBefore
    disk_free_after = $drive.AvailableFreeSpace
    timestamp_utc = [DateTime]::UtcNow.ToString('o')
}
$reportPath = Join-Path $workspaceRoot 'outputs\storage_cleanup\deletion_report.json'
$json = $report | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText($reportPath, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
$json
