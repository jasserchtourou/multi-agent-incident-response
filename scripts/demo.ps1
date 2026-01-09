# Demo script for Multi-Agent Incident Response System (PowerShell)

param(
    [Parameter(Position=0)]
    [string]$Command = "demo",
    
    [Parameter(Position=1)]
    [string]$FaultType = "error_rate",
    
    [Parameter(Position=2)]
    [int]$Duration = 60
)

Write-Host "`n🚀 Multi-Agent Incident Response Demo" -ForegroundColor Cyan
Write-Host "======================================`n" -ForegroundColor Cyan

function Check-Services {
    Write-Host "Checking services..." -ForegroundColor Blue
    
    try {
        $null = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -Method Get -TimeoutSec 5
    } catch {
        Write-Host "❌ Backend is not running. Start with: docker-compose up -d" -ForegroundColor Red
        exit 1
    }
    
    try {
        $null = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get -TimeoutSec 5
    } catch {
        Write-Host "❌ Demo service is not running. Start with: docker-compose up -d" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ All services are running`n" -ForegroundColor Green
}

function Trigger-Fault {
    param([string]$Type, [int]$Dur)
    
    Write-Host "⚡ Triggering fault: $Type for ${Dur}s..." -ForegroundColor Yellow
    
    $body = @{
        type = $Type
        duration_seconds = $Dur
    } | ConvertTo-Json
    
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/admin/fault" `
            -Method Post -ContentType "application/json" -Body $body
        $response | ConvertTo-Json -Depth 10
    } catch {
        Write-Host "Failed to trigger fault: $_" -ForegroundColor Red
    }
}

function Get-DemoStatus {
    Write-Host "`n📊 Demo Service Status:" -ForegroundColor Blue
    try {
        $status = Invoke-RestMethod -Uri "http://localhost:8000/api/demo/status" -Method Get
        $status | ConvertTo-Json -Depth 10
    } catch {
        Write-Host "Failed to get status: $_" -ForegroundColor Red
    }
}

function Get-Incidents {
    Write-Host "`n📋 Current Incidents:" -ForegroundColor Blue
    try {
        $incidents = Invoke-RestMethod -Uri "http://localhost:8000/api/incidents" -Method Get
        $incidents | ConvertTo-Json -Depth 10
    } catch {
        Write-Host "Failed to get incidents: $_" -ForegroundColor Red
    }
}

function Get-IncidentDetail {
    param([string]$Id)
    
    Write-Host "`n🔍 Incident Detail:" -ForegroundColor Blue
    try {
        $incident = Invoke-RestMethod -Uri "http://localhost:8000/api/incidents/$Id" -Method Get
        $incident | ConvertTo-Json -Depth 10
    } catch {
        Write-Host "Failed to get incident: $_" -ForegroundColor Red
    }
}

function Wait-ForDetection {
    Write-Host "`n⏳ Waiting for incident detection (up to 90 seconds)..." -ForegroundColor Yellow
    
    for ($i = 0; $i -lt 18; $i++) {
        Start-Sleep -Seconds 5
        Write-Host "." -NoNewline
        
        try {
            $incidents = Invoke-RestMethod -Uri "http://localhost:8000/api/incidents" -Method Get
            if ($incidents.total -gt 0) {
                Write-Host "`n✅ Incident detected!" -ForegroundColor Green
                return
            }
        } catch {
            # Continue waiting
        }
    }
    
    Write-Host "`n⚠️ No incident detected yet. Check the logs." -ForegroundColor Yellow
}

function Run-Demo {
    Check-Services
    
    Write-Host "Starting demo sequence...`n" -ForegroundColor Blue
    
    # 1. Check current status
    Get-DemoStatus
    
    # 2. Trigger a fault
    Trigger-Fault -Type "error_rate" -Dur 120
    
    # 3. Wait for detection
    Wait-ForDetection
    
    # 4. List incidents
    Get-Incidents
    
    Write-Host "`n🎉 Demo complete!" -ForegroundColor Green
    Write-Host "View the dashboard at: http://localhost:8000"
    Write-Host "View API docs at: http://localhost:8000/docs"
}

# Main switch
switch ($Command) {
    "demo" {
        Run-Demo
    }
    "fault" {
        Check-Services
        Trigger-Fault -Type $FaultType -Dur $Duration
    }
    "status" {
        Check-Services
        Get-DemoStatus
    }
    "incidents" {
        Check-Services
        Get-Incidents
    }
    "incident" {
        Check-Services
        Get-IncidentDetail -Id $FaultType
    }
    default {
        Write-Host "Usage: .\demo.ps1 {demo|fault [type] [duration]|status|incidents|incident [id]}"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  demo              Run full demo sequence"
        Write-Host "  fault [type]      Trigger a fault (error_rate, latency_spike, db_slow, memory_leak, dependency_down)"
        Write-Host "  status            Check demo service status"
        Write-Host "  incidents         List all incidents"
        Write-Host "  incident [id]     Get incident details"
    }
}

