# Master Script for All StaR Ablation Studies
# This script runs all ablation study scripts in sequence

Write-Host "`n$('=' * 80)" -ForegroundColor Cyan
Write-Host "StaR COMPREHENSIVE ABLATION STUDY" -ForegroundColor Cyan
Write-Host "$('=' * 80)" -ForegroundColor Cyan
Write-Host "This script will run the following ablation studies:" -ForegroundColor Yellow
Write-Host "1. Memory Dimension (32, 64, 128)" -ForegroundColor White
Write-Host "2. Memory Mechanism (GRU, LSTM, Transformer, MLP)" -ForegroundColor White
Write-Host "3. Message Passing (Enabled vs Disabled)" -ForegroundColor White
Write-Host "4. Time Dimension sensitivity" -ForegroundColor White
Write-Host "5. Message Dimension sensitivity" -ForegroundColor White
Write-Host "$('=' * 80)" -ForegroundColor Cyan

# Create master log directory
$masterLogsDir = "experiment_logs"
if (!(Test-Path $masterLogsDir)) {
    New-Item -ItemType Directory -Path $masterLogsDir | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$masterLog = "$masterLogsDir\master_ablation_study_$timestamp.log"

# Initialize master log
"=" * 80 | Out-File -FilePath $masterLog -Encoding utf8
"StaR COMPREHENSIVE ABLATION STUDY" | Out-File -FilePath $masterLog -Append -Encoding utf8
"Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $masterLog -Append -Encoding utf8
"=" * 80 | Out-File -FilePath $masterLog -Append -Encoding utf8
"`n" | Out-File -FilePath $masterLog -Append -Encoding utf8

# Track overall results
$totalStudies = 5
$completedStudies = 0
$failedStudies = @()

# Study 1: Memory Dimension Ablation
Write-Host "`n`n$('=' * 80)" -ForegroundColor Cyan
Write-Host "[1/$totalStudies] MEMORY DIMENSION ABLATION STUDY" -ForegroundColor Cyan
Write-Host "$('=' * 80)" -ForegroundColor Cyan
Write-Host "Testing memory dimensions: 32, 64, 128" -ForegroundColor Yellow
Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor Green

"`n[1/$totalStudies] MEMORY DIMENSION ABLATION STUDY" | Out-File -FilePath $masterLog -Append -Encoding utf8
"Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $masterLog -Append -Encoding utf8

$study1Start = Get-Date
try {
    & .\run_experiments_memory_dim.ps1
    if ($LASTEXITCODE -eq 0) {
        $completedStudies++
        $study1Duration = (Get-Date) - $study1Start
        Write-Host "`nMemory Dimension study completed successfully" -ForegroundColor Green
        Write-Host "Duration: $($study1Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        "Status: SUCCESS" | Out-File -FilePath $masterLog -Append -Encoding utf8
        "Duration: $($study1Duration.ToString('hh\:mm\:ss'))" | Out-File -FilePath $masterLog -Append -Encoding utf8
    }
    else {
        $failedStudies += "Memory Dimension Ablation"
        Write-Host "`nMemory Dimension study failed" -ForegroundColor Red
        "Status: FAILED" | Out-File -FilePath $masterLog -Append -Encoding utf8
    }
}
catch {
    $failedStudies += "Memory Dimension Ablation"
    Write-Host "`nMemory Dimension study encountered an error: $_" -ForegroundColor Red
    "Status: ERROR - $_" | Out-File -FilePath $masterLog -Append -Encoding utf8
}

# Study 2: Memory Mechanism Ablation
Write-Host "`n`n$('=' * 80)" -ForegroundColor Cyan
Write-Host "[2/$totalStudies] MEMORY MECHANISM ABLATION STUDY" -ForegroundColor Cyan
Write-Host "$('=' * 80)" -ForegroundColor Cyan
Write-Host "Testing memory mechanisms: GRU, LSTM, Transformer, MLP" -ForegroundColor Yellow
Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor Green

"`n[2/$totalStudies] MEMORY MECHANISM ABLATION STUDY" | Out-File -FilePath $masterLog -Append -Encoding utf8
"Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $masterLog -Append -Encoding utf8

$study2Start = Get-Date
try {
    & .\run_experiments_memory_mechanisms.ps1
    if ($LASTEXITCODE -eq 0) {
        $completedStudies++
        $study2Duration = (Get-Date) - $study2Start
        Write-Host "`nMemory Mechanism study completed successfully" -ForegroundColor Green
        Write-Host "Duration: $($study2Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        "Status: SUCCESS" | Out-File -FilePath $masterLog -Append -Encoding utf8
        "Duration: $($study2Duration.ToString('hh\:mm\:ss'))" | Out-File -FilePath $masterLog -Append -Encoding utf8
    }
    else {
        $failedStudies += "Memory Mechanism Ablation"
        Write-Host "`nMemory Mechanism study failed" -ForegroundColor Red
        "Status: FAILED" | Out-File -FilePath $masterLog -Append -Encoding utf8
    }
}
catch {
    $failedStudies += "Memory Mechanism Ablation"
    Write-Host "`nMemory Mechanism study encountered an error: $_" -ForegroundColor Red
    "Status: ERROR - $_" | Out-File -FilePath $masterLog -Append -Encoding utf8
}

# Study 3: Message Passing Ablation
Write-Host "`n`n$('=' * 80)" -ForegroundColor Cyan
Write-Host "[3/$totalStudies] MESSAGE PASSING ABLATION STUDY" -ForegroundColor Cyan
Write-Host "$('=' * 80)" -ForegroundColor Cyan
Write-Host "Testing message passing: Enabled vs Disabled" -ForegroundColor Yellow
Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor Green

"`n[3/$totalStudies] MESSAGE PASSING ABLATION STUDY" | Out-File -FilePath $masterLog -Append -Encoding utf8
"Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $masterLog -Append -Encoding utf8

$study3Start = Get-Date
try {
    & .\run_experiments_ablation_message_passing.ps1
    if ($LASTEXITCODE -eq 0) {
        $completedStudies++
        $study3Duration = (Get-Date) - $study3Start
        Write-Host "`nMessage Passing study completed successfully" -ForegroundColor Green
        Write-Host "Duration: $($study3Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        "Status: SUCCESS" | Out-File -FilePath $masterLog -Append -Encoding utf8
        "Duration: $($study3Duration.ToString('hh\:mm\:ss'))" | Out-File -FilePath $masterLog -Append -Encoding utf8
    }
    else {
        $failedStudies += "Message Passing Ablation"
        Write-Host "`nMessage Passing study failed" -ForegroundColor Red
        "Status: FAILED" | Out-File -FilePath $masterLog -Append -Encoding utf8
    }
}
catch {
    $failedStudies += "Message Passing Ablation"
    Write-Host "`nMessage Passing study encountered an error: $_" -ForegroundColor Red
    "Status: ERROR - $_" | Out-File -FilePath $masterLog -Append -Encoding utf8
}

# Study 4: Time Dimension Ablation
Write-Host "`n`n$('=' * 80)" -ForegroundColor Cyan
Write-Host "[4/$totalStudies] TIME DIMENSION ABLATION STUDY" -ForegroundColor Cyan
Write-Host "$('=' * 80)" -ForegroundColor Cyan
Write-Host "Testing time dimensions" -ForegroundColor Yellow
Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor Green

"`n[4/$totalStudies] TIME DIMENSION ABLATION STUDY" | Out-File -FilePath $masterLog -Append -Encoding utf8
"Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $masterLog -Append -Encoding utf8

$study4Start = Get-Date
try {
    & .\run_experiments_time_dim.ps1
    if ($LASTEXITCODE -eq 0) {
        $completedStudies++
        $study4Duration = (Get-Date) - $study4Start
        Write-Host "`nTime Dimension study completed successfully" -ForegroundColor Green
        Write-Host "Duration: $($study4Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        "Status: SUCCESS" | Out-File -FilePath $masterLog -Append -Encoding utf8
        "Duration: $($study4Duration.ToString('hh\:mm\:ss'))" | Out-File -FilePath $masterLog -Append -Encoding utf8
    }
    else {
        $failedStudies += "Time Dimension Ablation"
        Write-Host "`nTime Dimension study failed" -ForegroundColor Red
        "Status: FAILED" | Out-File -FilePath $masterLog -Append -Encoding utf8
    }
}
catch {
    $failedStudies += "Time Dimension Ablation"
    Write-Host "`nTime Dimension study encountered an error: $_" -ForegroundColor Red
    "Status: ERROR - $_" | Out-File -FilePath $masterLog -Append -Encoding utf8
}

# Study 5: Message Dimension Ablation
Write-Host "`n`n$('=' * 80)" -ForegroundColor Cyan
Write-Host "[5/$totalStudies] MESSAGE DIMENSION ABLATION STUDY" -ForegroundColor Cyan
Write-Host "$('=' * 80)" -ForegroundColor Cyan
Write-Host "Testing message dimensions" -ForegroundColor Yellow
Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor Green

"`n[5/$totalStudies] MESSAGE DIMENSION ABLATION STUDY" | Out-File -FilePath $masterLog -Append -Encoding utf8
"Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $masterLog -Append -Encoding utf8

$study5Start = Get-Date
try {
    & .\run_experiments_message_dim.ps1
    if ($LASTEXITCODE -eq 0) {
        $completedStudies++
        $study5Duration = (Get-Date) - $study5Start
        Write-Host "`nMessage Dimension study completed successfully" -ForegroundColor Green
        Write-Host "Duration: $($study5Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
        "Status: SUCCESS" | Out-File -FilePath $masterLog -Append -Encoding utf8
        "Duration: $($study5Duration.ToString('hh\:mm\:ss'))" | Out-File -FilePath $masterLog -Append -Encoding utf8
    }
    else {
        $failedStudies += "Message Dimension Ablation"
        Write-Host "`nMessage Dimension study failed" -ForegroundColor Red
        "Status: FAILED" | Out-File -FilePath $masterLog -Append -Encoding utf8
    }
}
catch {
    $failedStudies += "Message Dimension Ablation"
    Write-Host "`nMessage Dimension study encountered an error: $_" -ForegroundColor Red
    "Status: ERROR - $_" | Out-File -FilePath $masterLog -Append -Encoding utf8
}

# Final summary
Write-Host "`n`n$('=' * 80)" -ForegroundColor Cyan
Write-Host "ALL ABLATION STUDIES COMPLETED" -ForegroundColor Cyan
Write-Host "$('=' * 80)" -ForegroundColor Cyan

"`n`n" + "=" * 80 | Out-File -FilePath $masterLog -Append -Encoding utf8
"FINAL SUMMARY - ALL ABLATION STUDIES" | Out-File -FilePath $masterLog -Append -Encoding utf8
"=" * 80 | Out-File -FilePath $masterLog -Append -Encoding utf8
"Finished at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $masterLog -Append -Encoding utf8
"Total studies: $totalStudies" | Out-File -FilePath $masterLog -Append -Encoding utf8
"Completed successfully: $completedStudies" | Out-File -FilePath $masterLog -Append -Encoding utf8
"Failed: $($failedStudies.Count)" | Out-File -FilePath $masterLog -Append -Encoding utf8

Write-Host "Total studies: $totalStudies"
Write-Host "Completed successfully: $completedStudies" -ForegroundColor Green
Write-Host "Failed: $($failedStudies.Count)" -ForegroundColor $(if ($failedStudies.Count -gt 0) { "Red" } else { "Green" })

if ($failedStudies.Count -gt 0) {
    Write-Host "`nFailed studies:" -ForegroundColor Red
    "`nFailed studies:" | Out-File -FilePath $masterLog -Append -Encoding utf8
    foreach ($failed in $failedStudies) {
        Write-Host "  - $failed" -ForegroundColor Red
        "  - $failed" | Out-File -FilePath $masterLog -Append -Encoding utf8
    }
}

Write-Host "`nAll study logs saved in: $masterLogsDir" -ForegroundColor Yellow
Write-Host "Master summary log: $masterLog" -ForegroundColor Yellow
Write-Host "$('=' * 80)`n" -ForegroundColor Cyan

"`n" + "=" * 80 | Out-File -FilePath $masterLog -Append -Encoding utf8
"ABLATION STUDIES OVERVIEW:" | Out-File -FilePath $masterLog -Append -Encoding utf8
"1. Memory Dimension: Tests impact of memory vector size (32, 64, 128)" | Out-File -FilePath $masterLog -Append -Encoding utf8
"2. Memory Mechanism: Compares GRU, LSTM, Transformer, MLP updaters" | Out-File -FilePath $masterLog -Append -Encoding utf8
"3. Message Passing: Tests temporal mechanism contribution (enabled vs disabled)" | Out-File -FilePath $masterLog -Append -Encoding utf8
"4. Time Dimension: Tests impact of temporal encoding dimension" | Out-File -FilePath $masterLog -Append -Encoding utf8
"5. Message Dimension: Tests impact of message dimension" | Out-File -FilePath $masterLog -Append -Encoding utf8
"=" * 80 | Out-File -FilePath $masterLog -Append -Encoding utf8

# Exit with appropriate code
if ($failedStudies.Count -gt 0) {
    exit 1
}
else {
    exit 0
}
