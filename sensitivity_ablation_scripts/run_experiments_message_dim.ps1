# Message Dimension Ablation Study Script for StaR Models
# This script tests different message passing dimension configurations

# Create logs directory if it doesn't exist
$logsDir = "experiment_logs"
if (!(Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
    Write-Host "Created directory: $logsDir"
}

# Get timestamp for this run
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$summaryLog = "$logsDir\message_dim_summary_$timestamp.log"

# Initialize summary log
"=" * 80 | Out-File -FilePath $summaryLog -Encoding utf8
"StaR Message Dimension Ablation Study" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"=" * 80 | Out-File -FilePath $summaryLog -Append -Encoding utf8
"`n" | Out-File -FilePath $summaryLog -Append -Encoding utf8

# Define message dimensions to test
$messageDims = @(8, 16, 32, 64)

# Define datasets to test (focusing on random_connection with various probabilities)
$datasets = @(
    @{
        Name = "random_connection_0.45"
        Command = "--dataset_name random_connection --base_connection_prob 0.45 --preprocessing_data 1"
    },
    @{
        Name = "random_connection_0.55"
        Command = "--dataset_name random_connection --base_connection_prob 0.55 --preprocessing_data 1"
    },
    @{
        Name = "random_connection_0.65"
        Command = "--dataset_name random_connection --base_connection_prob 0.65 --preprocessing_data 1"
    },
    @{
        Name = "random_connection_0.75"
        Command = "--dataset_name random_connection --base_connection_prob 0.75 --preprocessing_data 1"
    },
    @{
        Name = "random_connection_0.85"
        Command = "--dataset_name random_connection --base_connection_prob 0.85 --preprocessing_data 1"
    }
)

# Build complete command list
$commands = @()
foreach ($dataset in $datasets) {
    foreach ($msgDim in $messageDims) {
        $commands += @{
            Name = "$($dataset.Name)_msg$msgDim"
            Command = "uv run python main_tgn_original.py $($dataset.Command) --message_dim $msgDim"
            LogFile = "$logsDir\message_ablation_$($dataset.Name)_dim${msgDim}_$timestamp.log"
            Dataset = $dataset.Name
            MessageDim = $msgDim
        }
    }
}

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
    Write-Host "Dataset: $($cmd.Dataset)" -ForegroundColor Yellow
    Write-Host "Message Dimension: $($cmd.MessageDim)" -ForegroundColor Yellow
    Write-Host "Command: $($cmd.Command)" -ForegroundColor Gray
    Write-Host "Log file: $($cmd.LogFile)" -ForegroundColor Gray
    Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n" -ForegroundColor Green
    
    # Log to summary
    "`n[$cmdNum/$totalCommands] Running: $($cmd.Name)" | Out-File -FilePath $summaryLog -Append -Encoding utf8
    "Dataset: $($cmd.Dataset)" | Out-File -FilePath $summaryLog -Append -Encoding utf8
    "Message Dimension: $($cmd.MessageDim)" | Out-File -FilePath $summaryLog -Append -Encoding utf8
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
Write-Host "MESSAGE DIMENSION ABLATION STUDY COMPLETED" -ForegroundColor Cyan
Write-Host "$('=' * 80)" -ForegroundColor Cyan

"`n`n" + "=" * 80 | Out-File -FilePath $summaryLog -Append -Encoding utf8
"FINAL SUMMARY - MESSAGE DIMENSION ABLATION" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"=" * 80 | Out-File -FilePath $summaryLog -Append -Encoding utf8
"Finished at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"Total configurations tested: $totalCommands" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"Successful: $successCount" | Out-File -FilePath $summaryLog -Append -Encoding utf8
"Failed: $($failedCommands.Count)" | Out-File -FilePath $summaryLog -Append -Encoding utf8

Write-Host "Total configurations tested: $totalCommands"
Write-Host "Message dimensions: $($messageDims -join ', ')" -ForegroundColor Magenta
Write-Host "Datasets: $($datasets.Name -join ', ')" -ForegroundColor Magenta
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

