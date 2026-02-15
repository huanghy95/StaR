# Sequential Training Script for Random Connection Ablation Study
# This script tests StaR-GC with GRU memory disabled (--disable_message_passing)
# on random_connection datasets with different connection probabilities
# Note: Message passing is KEPT to respect dynamic graph structure (fair comparison)

# Create logs directory if it doesn't exist
$logsDir = "experiment_logs"
if (!(Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
    Write-Host "Created directory: $logsDir"
}

# Get timestamp for this run
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$summaryLog = "$logsDir\summary_random_ablation_$timestamp.log"

# Initialize summary log
"=" * 80 | Out-File -FilePath $summaryLog -Encoding utf8
"Random Connection Ablation Study (No GRU Memory)" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"Testing connection probabilities: 0.45, 0.55, 0.65, 0.75, 0.85" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"=" * 80 | Out-File -FilePath $summaryLog -Append -Encoding utf8
"`n" | Out-File -FilePath $summaryLog -Append -Encoding utf8

# Define commands to run in order - Random Connection Ablation Study (No GRU Memory)
# This ablation keeps message passing (respects adjacency_mask) but removes GRU temporal memory
# Testing different connection probabilities to evaluate dynamic graph handling
$commands = @(
    @{
        Name = "random_connection_0.45"
        Command = "uv run python main_tgn_original.py --dataset_name random_connection --base_connection_prob 0.45 --preprocessing_data 1 --disable_message_passing"
        LogFile = "$logsDir\random_ablation_exp1_p0.45_$timestamp.log"
    },
    @{
        Name = "random_connection_0.55"
        Command = "uv run python main_tgn_original.py --dataset_name random_connection --base_connection_prob 0.55 --preprocessing_data 1 --disable_message_passing"
        LogFile = "$logsDir\random_ablation_exp2_p0.55_$timestamp.log"
    },
    @{
        Name = "random_connection_0.65"
        Command = "uv run python main_tgn_original.py --dataset_name random_connection --base_connection_prob 0.65 --preprocessing_data 1 --disable_message_passing"
        LogFile = "$logsDir\random_ablation_exp3_p0.65_$timestamp.log"
    },
    @{
        Name = "random_connection_0.75"
        Command = "uv run python main_tgn_original.py --dataset_name random_connection --base_connection_prob 0.75 --preprocessing_data 1 --disable_message_passing"
        LogFile = "$logsDir\random_ablation_exp4_p0.75_$timestamp.log"
    },
    @{
        Name = "random_connection_0.85"
        Command = "uv run python main_tgn_original.py --dataset_name random_connection --base_connection_prob 0.85 --preprocessing_data 1 --disable_message_passing"
        LogFile = "$logsDir\random_ablation_exp5_p0.85_$timestamp.log"
    }
)

# Track results
$totalCommands = $commands.Count
$successCount = 0
$failedCommands = @()

# Run each command sequentially
for ($i = 0; $i -lt $commands.Count; $i++) {
    $cmd = $commands[$i]
    $cmdNum = $i + 1
    
    Write-Host "`n$('=' * 80)" -ForegroundColor Cyan
    Write-Host "[$cmdNum/$totalCommands] Running: $($cmd.Name)" -ForegroundColor Cyan
    Write-Host "$('=' * 80)" -ForegroundColor Cyan
    Write-Host "Command: $($cmd.Command)" -ForegroundColor Yellow
    Write-Host "Log file: $($cmd.LogFile)" -ForegroundColor Yellow
    Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor Green
    
    # Log to summary
    "`n[$cmdNum/$totalCommands] Running: $($cmd.Name)" | Out-File -FilePath $summaryLog -Append -Encoding utf8
    "Command: $($cmd.Command)" | Out-File -FilePath $summaryLog -Append -Encoding utf8
    "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $summaryLog -Append -Encoding utf8
    
    # Record start time
    $startTime = Get-Date
    
    # Run command and capture output to log file
    # Using *>&1 to capture all output streams (stdout and stderr)
    try {
        # Execute command and redirect all output to log file
        $output = Invoke-Expression $cmd.Command 2>&1 | Tee-Object -FilePath $cmd.LogFile -Append
        
        # Check exit code
        if ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE) {
            $endTime = Get-Date
            $duration = $endTime - $startTime
            
            Write-Host "`n$('=' * 80)" -ForegroundColor Green
            Write-Host "✓ SUCCESS: $($cmd.Name) completed" -ForegroundColor Green
            Write-Host "Duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
            Write-Host "$('=' * 80)`n" -ForegroundColor Green
            
            "Status: SUCCESS" | Out-File -FilePath $summaryLog -Append -Encoding utf8
            "Completed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $summaryLog -Append -Encoding utf8
            "Duration: $($duration.ToString('hh\:mm\:ss'))" | Out-File -FilePath $summaryLog -Append -Encoding utf8
            
            $successCount++
        }
        else {
            $endTime = Get-Date
            $duration = $endTime - $startTime
            
            Write-Host "`n$('=' * 80)" -ForegroundColor Red
            Write-Host "✗ ERROR: $($cmd.Name) failed with exit code $LASTEXITCODE" -ForegroundColor Red
            Write-Host "Duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Red
            Write-Host "Check log file for details: $($cmd.LogFile)" -ForegroundColor Red
            Write-Host "$('=' * 80)`n" -ForegroundColor Red
            
            "Status: FAILED (Exit code: $LASTEXITCODE)" | Out-File -FilePath $summaryLog -Append -Encoding utf8
            "Completed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $summaryLog -Append -Encoding utf8
            "Duration: $($duration.ToString('hh\:mm\:ss'))" | Out-File -FilePath $summaryLog -Append -Encoding utf8
            
            $failedCommands += $cmd.Name
        }
    }
    catch {
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        Write-Host "`n$('=' * 80)" -ForegroundColor Red
        Write-Host "✗ EXCEPTION: $($cmd.Name) encountered an error" -ForegroundColor Red
        Write-Host "Error: $_" -ForegroundColor Red
        Write-Host "Duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Red
        Write-Host "Check log file for details: $($cmd.LogFile)" -ForegroundColor Red
        Write-Host "$('=' * 80)`n" -ForegroundColor Red
        
        "Status: EXCEPTION" | Out-File -FilePath $summaryLog -Append -Encoding utf8
        "Error: $_" | Out-File -FilePath $summaryLog -Append -Encoding utf8
        "Completed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $summaryLog -Append -Encoding utf8
        "Duration: $($duration.ToString('hh\:mm\:ss'))" | Out-File -FilePath $summaryLog -Append -Encoding utf8
        
        $failedCommands += $cmd.Name
    }
}

# Final summary
Write-Host "`n`n$('=' * 80)" -ForegroundColor Cyan
Write-Host "ALL RANDOM CONNECTION ABLATION EXPERIMENTS COMPLETED" -ForegroundColor Cyan
Write-Host "$('=' * 80)" -ForegroundColor Cyan

"`n`n" + "=" * 80 | Out-File -FilePath $summaryLog -Append -Encoding utf8
"FINAL SUMMARY" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"=" * 80 | Out-File -FilePath $summaryLog -Append -Encoding utf8
"Finished at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"Total commands: $totalCommands" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"Successful: $successCount" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"Failed: $($failedCommands.Count)" | Out-File -FilePath $summaryLog -Append -Encoding utf8

Write-Host "Total commands: $totalCommands"
Write-Host "Successful: $successCount" -ForegroundColor Green
Write-Host "Failed: $($failedCommands.Count)" -ForegroundColor $(if ($failedCommands.Count -gt 0) { "Red" } else { "Green" })

if ($failedCommands.Count -gt 0) {
    Write-Host "`nFailed experiments:" -ForegroundColor Red
    "`nFailed experiments:" | Out-File -FilePath $summaryLog -Append -Encoding utf8
    foreach ($failed in $failedCommands) {
        Write-Host "  - $failed" -ForegroundColor Red
        "  - $failed" | Out-File -FilePath $summaryLog -Append -Encoding utf8
    }
}

Write-Host "`nAll logs saved in: $logsDir" -ForegroundColor Yellow
Write-Host "Summary log: $summaryLog" -ForegroundColor Yellow
Write-Host "$('=' * 80)`n" -ForegroundColor Cyan

"=" * 80 | Out-File -FilePath $summaryLog -Append -Encoding utf8

# Exit with appropriate code
if ($failedCommands.Count -gt 0) {
    exit 1
}
else {
    exit 0
}


