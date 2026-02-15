# Reverse Ablation Study: Capacity-Matched AERCA
# This script runs ORIGINAL AERCA with INCREASED CAPACITY to match StaR
# 
# Hypothesis: If we give original AERCA the same capacity as StaR,
#             they should perform similarly.
#
# This proves that StaR's advantage is CAPACITY, not temporal mechanisms.

# Create logs directory if it doesn't exist
$logsDir = "experiment_logs"
if (!(Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
    Write-Host "Created directory: $logsDir"
}

# Get timestamp for this run
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$summaryLog = "$logsDir\summary_capacity_matched_aerca_$timestamp.log"

# Initialize summary log
"=" * 80 | Out-File -FilePath $summaryLog -Encoding utf8
"Capacity-Matched AERCA Experiments (Reverse Ablation)" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"=" * 80 | Out-File -FilePath $summaryLog -Append -Encoding utf8
"`n" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"HYPOTHESIS:" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"  Original AERCA with matched capacity ≈ StaR performance" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"`n" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"APPROACH:" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"  Increase AERCA's hidden_layer_size to match StaR parameter count" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"`n" | Out-File -FilePath $summaryLog -Append -Encoding utf8

# Define commands to run
# Using main.py (original AERCA) with increased --hidden_layer_size
$commands = @(
    @{
        Name = "linear_capacity_matched"
        Command = "uv run python main.py --dataset_name linear --hidden_layer_size 873"
        LogFile = "$logsDir\capacity_matched_exp1_linear_$timestamp.log"
        Description = "Linear dataset with AERCA hidden_size=873 (matches StaR's 9,600 params)"
        Comparison = "Compare to: StaR with default hidden_size=128"
    },
    @{
        Name = "msds_capacity_matched"
        Command = "uv run python main.py --dataset_name msds --preprocessing_data 1 --hidden_layer_size 339"
        LogFile = "$logsDir\capacity_matched_exp2_msds_$timestamp.log"
        Description = "MSDS dataset with AERCA hidden_size=339 (matches StaR's 13,184 params)"
        Comparison = "Compare to: StaR with default hidden_size=128"
    },
    @{
        Name = "nonlinear_capacity_matched"
        Command = "uv run python main.py --dataset_name nonlinear --hidden_layer_size 873"
        LogFile = "$logsDir\capacity_matched_exp3_nonlinear_$timestamp.log"
        Description = "Nonlinear dataset with AERCA hidden_size=873 (matches StaR's 9,600 params)"
        Comparison = "Compare to: StaR with default hidden_size=128"
    },
    @{
        Name = "random_connection_0.65_capacity_matched"
        Command = "uv run python main.py --dataset_name random_connection --base_connection_prob 0.65 --preprocessing_data 1 --hidden_layer_size 873"
        LogFile = "$logsDir\capacity_matched_exp4_random_0.65_$timestamp.log"
        Description = "Random Connection (p=0.65) with AERCA hidden_size=873 (matches StaR's 9,600 params)"
        Comparison = "Compare to: StaR with default hidden_size=128"
    },
    @{
        Name = "random_connection_0.45_capacity_matched"
        Command = "uv run python main.py --dataset_name random_connection --base_connection_prob 0.45 --preprocessing_data 1 --hidden_layer_size 873"
        LogFile = "$logsDir\capacity_matched_exp5_random_0.45_$timestamp.log"
        Description = "Random Connection (p=0.45) with AERCA hidden_size=873"
        Comparison = "Compare to: StaR ablation results"
    },
    @{
        Name = "random_connection_0.85_capacity_matched"
        Command = "uv run python main.py --dataset_name random_connection --base_connection_prob 0.85 --preprocessing_data 1 --hidden_layer_size 873"
        LogFile = "$logsDir\capacity_matched_exp6_random_0.85_$timestamp.log"
        Description = "Random Connection (p=0.85) with AERCA hidden_size=873"
        Comparison = "Compare to: StaR ablation results"
    },
    @{
        Name = "random_connection_0.55_capacity_matched"
        Command = "uv run python main.py --dataset_name random_connection --base_connection_prob 0.55 --preprocessing_data 1 --hidden_layer_size 873"
        LogFile = "$logsDir\capacity_matched_exp7_random_0.55_$timestamp.log"
        Description = "Random Connection (p=0.55) with AERCA hidden_size=873"
        Comparison = "Compare to: StaR ablation results"
    },
    @{
        Name = "random_connection_0.75_capacity_matched"
        Command = "uv run python main.py --dataset_name random_connection --base_connection_prob 0.75 --preprocessing_data 1 --hidden_layer_size 873"
        LogFile = "$logsDir\capacity_matched_exp8_random_0.75_$timestamp.log"
        Description = "Random Connection (p=0.75) with AERCA hidden_size=873"
        Comparison = "Compare to: StaR ablation results"
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
    Write-Host "Description: $($cmd.Description)" -ForegroundColor Yellow
    Write-Host "Command: $($cmd.Command)" -ForegroundColor Yellow
    Write-Host "Log file: $($cmd.LogFile)" -ForegroundColor Yellow
    Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor Green
    
    # Log to summary
    "`n[$cmdNum/$totalCommands] $($cmd.Name)" | Out-File -FilePath $summaryLog -Append -Encoding utf8
    "Description: $($cmd.Description)" | Out-File -FilePath $summaryLog -Append -Encoding utf8
    "Command: $($cmd.Command)" | Out-File -FilePath $summaryLog -Append -Encoding utf8
    "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $summaryLog -Append -Encoding utf8
    
    # Record start time
    $startTime = Get-Date
    
    # Run command and capture output to log file
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
Write-Host "ALL CAPACITY-MATCHED AERCA EXPERIMENTS COMPLETED" -ForegroundColor Cyan
Write-Host "$('=' * 80)" -ForegroundColor Cyan

"`n`n" + "=" * 80 | Out-File -FilePath $summaryLog -Append -Encoding utf8
"FINAL SUMMARY - REVERSE ABLATION STUDY" | Out-File -FilePath $summaryLog -Append -Encoding utf8
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

# Print interpretation guide
Write-Host "`n$('=' * 80)" -ForegroundColor Magenta
Write-Host "INTERPRETATION GUIDE" -ForegroundColor Magenta
Write-Host "$('=' * 80)" -ForegroundColor Magenta
Write-Host ""
Write-Host "Expected Results (if hypothesis is correct):" -ForegroundColor Yellow
Write-Host "  - Capacity-matched AERCA ≈ StaR (ablation)" -ForegroundColor White
Write-Host "  - Capacity-matched AERCA ≈ StaR (normal)" -ForegroundColor White
Write-Host "  - Capacity-matched AERCA >> Original AERCA (default size)" -ForegroundColor White
Write-Host ""
Write-Host "This would prove:" -ForegroundColor Yellow
Write-Host "  ✓ StaR's advantage = CAPACITY" -ForegroundColor Green
Write-Host "  ✓ Temporal mechanisms = MINIMAL contribution" -ForegroundColor Green
Write-Host "  ✓ Simpler alternatives (Wide-AERCA) work just as well" -ForegroundColor Green
Write-Host ""
Write-Host "Compare these results to:" -ForegroundColor Yellow
Write-Host "  1. run_experiments_linear_sequential.ps1 (StaR with ablation)" -ForegroundColor White
Write-Host "  2. Original AERCA results (default hidden_layer_size)" -ForegroundColor White
Write-Host "$('=' * 80)`n" -ForegroundColor Magenta

# Add interpretation to summary log
"`n" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"=" * 80 | Out-File -FilePath $summaryLog -Append -Encoding utf8
"INTERPRETATION GUIDE" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"=" * 80 | Out-File -FilePath $summaryLog -Append -Encoding utf8
"`nExpected Results (if hypothesis is correct):" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"  - Capacity-matched AERCA ≈ StaR (ablation)" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"  - Capacity-matched AERCA ≈ StaR (normal)" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"  - Capacity-matched AERCA >> Original AERCA (default size)" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"`nThis would prove:" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"  ✓ StaR's advantage = CAPACITY" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"  ✓ Temporal mechanisms = MINIMAL contribution" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"  ✓ Simpler alternatives (Wide-AERCA) work just as well" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"=" * 80 | Out-File -FilePath $summaryLog -Append -Encoding utf8

# Exit with appropriate code
if ($failedCommands.Count -gt 0) {
    exit 1
}
else {
    exit 0
}

