param(
    [Parameter(Mandatory=$true)][ValidateSet('CheckDuplicates','DeleteDuplicates','DeleteArchived')][string]$Mode,
    [Parameter(Mandatory=$true)][string]$Plan,
    [Parameter(Mandatory=$true)][string]$ExpectedPlanSha256
)
$ErrorActionPreference = 'Stop'
$WorkspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$PlanFile = (Resolve-Path -LiteralPath $Plan).ProviderPath
if ((Get-FileHash -LiteralPath $PlanFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedPlanSha256.ToLowerInvariant()) { throw 'Cleanup plan changed; stop.' }
$Spec = Get-Content -LiteralPath $PlanFile -Raw -Encoding UTF8 | ConvertFrom-Json
if ([IO.Path]::GetFullPath($Spec.root).TrimEnd('\') -ne $WorkspaceRoot) { throw 'Wrong workspace.' }
$AuditDir = Split-Path -Parent $PlanFile
$Journal = Join-Path $AuditDir ('EXECUTED_' + $Mode + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.jsonl')

function Get-SafePath([string]$Relative) {
    if ([IO.Path]::IsPathRooted($Relative)) { throw 'Absolute plan member forbidden.' }
    $Full = [IO.Path]::GetFullPath((Join-Path $WorkspaceRoot $Relative))
    if (-not $Full.StartsWith($WorkspaceRoot + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Target escaped workspace.' }
    if (Test-Path -LiteralPath $Full) {
        $Resolved = (Resolve-Path -LiteralPath $Full).ProviderPath
        if (-not $Resolved.StartsWith($WorkspaceRoot + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Resolved target escaped workspace.' }
        $Walk = $Full
        while ($Walk.Length -ge $WorkspaceRoot.Length) {
            if (((Get-Item -LiteralPath $Walk -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Reparse/symlink ancestor forbidden.' }
            if ($Walk -eq $WorkspaceRoot) { break }
            $Walk = Split-Path -Parent $Walk
        }
    }
    return $Full
}
function Assert-Unprotected([string]$Relative,[bool]$CacheException=$false) {
    foreach ($Hot in $Spec.protected_prefixes) {
        if ($Relative -eq $Hot -or $Relative.StartsWith($Hot+'/') -or $Hot.StartsWith($Relative+'/')) {
            $ExactCache = 'outputs/stage_k/s1000_test100_phase1_v1/METRIC_CACHE/models--FacebookAI--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59/pytorch_model.bin'
            if (-not ($CacheException -and $Relative -eq $ExactCache)) { throw ('Protected scientific path: '+$Relative) }
        }
    }
}
function Assert-Hash([string]$Path,[string]$Expected,[long]$Size) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw ('Required file missing: '+$Path) }
    if ((Get-Item -LiteralPath $Path).Length -ne $Size -or (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected) { throw ('Size/SHA mismatch: '+$Path) }
}
function Log-Event($Object) { $Object | ConvertTo-Json -Compress -Depth 6 | Add-Content -LiteralPath $Journal -Encoding UTF8 }

if ($Mode -in @('CheckDuplicates','DeleteDuplicates')) {
    $Approved = @()
    foreach ($Entry in $Spec.duplicate_deletions) {
        Assert-Unprotected $Entry.path ([bool]$Entry.operational_protected_exception)
        $Target=Get-SafePath $Entry.path; $Keep=Get-SafePath $Entry.keep
        if ($Target -eq $Keep) { throw 'Target equals keeper.' }
        if (-not (Test-Path -LiteralPath $Target)) { Write-Host ('Already absent: '+$Entry.path); continue }
        Assert-Hash $Target $Entry.sha256 $Entry.bytes
        Assert-Hash $Keep $Entry.sha256 $Entry.bytes
        $Approved += $Entry
    }
    Write-Host ('Current duplicate/keeper verification PASS: '+$Approved.Count+' files.')
    if ($Mode -eq 'DeleteDuplicates') {
        foreach ($Entry in $Approved) {
            # Recheck immediately; never enumerate in another shell and pass paths to cmd /c.
            $Target=Get-SafePath $Entry.path; $Keep=Get-SafePath $Entry.keep
            Assert-Hash $Target $Entry.sha256 $Entry.bytes; Assert-Hash $Keep $Entry.sha256 $Entry.bytes
            Remove-Item -LiteralPath $Target -Force
            Log-Event @{action='deleted_exact_duplicate';target=$Target;keeper=$Keep;sha256=$Entry.sha256;UTC=[DateTime]::UtcNow.ToString('o')}
        }
    }
    exit 0
}

# Cold roots can be removed ONLY after the independent HF uploader's full remote verification.
foreach ($Group in $Spec.retired_archive_groups) {
    Assert-Unprotected $Group.path
    $Target=Get-SafePath $Group.path
    if (-not (Test-Path -LiteralPath $Target)) { Write-Host ('Already absent: '+$Group.path); continue }
    $ReceiptFile=Join-Path $AuditDir ('HF_RECEIPTS/'+$Group.id+'.json')
    if (-not (Test-Path -LiteralPath $ReceiptFile)) { throw ('No verified HF backup; source kept: '+$Group.path) }
    $Receipt=Get-Content -LiteralPath $ReceiptFile -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($Receipt.status -ne 'PASS_FULL_REMOTE_ARCHIVE_AND_EVERY_MEMBER_SHA256' -or $Receipt.group_id -ne $Group.id -or $Receipt.source_path -ne $Group.path) { throw 'Wrong or incomplete remote backup receipt.' }
    $Python=Join-Path $WorkspaceRoot '.venv-caption-pilot/Scripts/python.exe'
    & $Python -X utf8 -B (Join-Path $PSScriptRoot 'archive_retired_runs_private_hf_20261029_v1.py') --plan $PlanFile --plan-sha $ExpectedPlanSha256 --check-remote --group $Group.id
    if ($LASTEXITCODE -ne 0) { throw 'Remote pinned backup is unavailable/changed; source kept.' }
    $ManifestFile=Join-Path $AuditDir ('HF_MANIFESTS/'+$Group.id+'.json')
    if ((Get-FileHash -LiteralPath $ManifestFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Receipt.manifest_sha256) { throw 'Local archive manifest changed.' }
    $Ledger=Get-Content -LiteralPath $ManifestFile -Raw -Encoding UTF8 | ConvertFrom-Json
    $Actual=@(Get-ChildItem -LiteralPath $Target -Recurse -Force)
    if (@($Actual | Where-Object {($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0}).Count -ne 0) { throw 'Link in recursive target; kept.' }
    $ActualFiles=@($Actual | Where-Object {-not $_.PSIsContainer})
    if ($ActualFiles.Count -ne @($Ledger.files).Count) { throw 'Source tree changed after backup; kept.' }
    foreach ($Entry in $Ledger.files) {
        if (-not $Entry.path.StartsWith($Group.path+'/',[StringComparison]::OrdinalIgnoreCase)) { throw 'Archive entry outside named source tree.' }
        Assert-Hash (Get-SafePath $Entry.path) $Entry.sha256 $Entry.bytes
    }
    # Fully resolved exact named subtree verified within workspace, no protected descendants.
    Remove-Item -LiteralPath $Target -Recurse -Force
    Log-Event @{action='deleted_HF_verified_retired_tree';target=$Target;manifest_sha256=$Receipt.manifest_sha256;archive_sha256=$Receipt.archive_sha256;HF_revision=$Receipt.revision;UTC=[DateTime]::UtcNow.ToString('o')}
    if ($Receipt.local_archive_relative) {
        $Staged=Get-SafePath $Receipt.local_archive_relative
        $StagingRoot=[IO.Path]::GetFullPath((Join-Path $AuditDir 'UPLOAD_STAGING')).TrimEnd('\')+'\'
        if (-not $Staged.StartsWith($StagingRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected local staging archive location.' }
        if (Test-Path -LiteralPath $Staged) { Assert-Hash $Staged $Receipt.archive_sha256 $Receipt.archive_bytes; Remove-Item -LiteralPath $Staged -Force }
    }
}
