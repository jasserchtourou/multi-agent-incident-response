#!/bin/bash
# Demo script for Multi-Agent Incident Response System

set -e

echo "🚀 Multi-Agent Incident Response Demo"
echo "======================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if services are running
check_services() {
    echo -e "${BLUE}Checking services...${NC}"
    
    if ! curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo -e "${RED}❌ Backend is not running. Start with: docker-compose up -d${NC}"
        exit 1
    fi
    
    if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "${RED}❌ Demo service is not running. Start with: docker-compose up -d${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All services are running${NC}"
}

# Trigger a fault
trigger_fault() {
    local fault_type=${1:-"error_rate"}
    local duration=${2:-60}
    
    echo ""
    echo -e "${YELLOW}⚡ Triggering fault: ${fault_type} for ${duration}s...${NC}"
    
    response=$(curl -s -X POST "http://localhost:8000/api/admin/fault" \
        -H "Content-Type: application/json" \
        -d "{\"type\": \"${fault_type}\", \"duration_seconds\": ${duration}}")
    
    echo "$response" | python -m json.tool 2>/dev/null || echo "$response"
}

# Check demo status
check_status() {
    echo ""
    echo -e "${BLUE}📊 Demo Service Status:${NC}"
    curl -s http://localhost:8000/api/demo/status | python -m json.tool 2>/dev/null
}

# List incidents
list_incidents() {
    echo ""
    echo -e "${BLUE}📋 Current Incidents:${NC}"
    curl -s http://localhost:8000/api/incidents | python -m json.tool 2>/dev/null
}

# Get incident detail
get_incident() {
    local id=$1
    echo ""
    echo -e "${BLUE}🔍 Incident Detail:${NC}"
    curl -s "http://localhost:8000/api/incidents/${id}" | python -m json.tool 2>/dev/null
}

# Wait for detection
wait_for_detection() {
    echo ""
    echo -e "${YELLOW}⏳ Waiting for incident detection (up to 90 seconds)...${NC}"
    
    for i in {1..18}; do
        sleep 5
        count=$(curl -s http://localhost:8000/api/incidents | python -c "import sys,json; data=json.load(sys.stdin); print(data.get('total', 0))" 2>/dev/null || echo "0")
        echo -n "."
        if [ "$count" -gt "0" ]; then
            echo ""
            echo -e "${GREEN}✅ Incident detected!${NC}"
            return 0
        fi
    done
    
    echo ""
    echo -e "${YELLOW}⚠️ No incident detected yet. Check the logs.${NC}"
}

# Main demo flow
run_demo() {
    check_services
    
    echo ""
    echo -e "${BLUE}Starting demo sequence...${NC}"
    
    # 1. Check current status
    check_status
    
    # 2. Trigger a fault
    trigger_fault "error_rate" 120
    
    # 3. Wait for detection
    wait_for_detection
    
    # 4. List incidents
    list_incidents
    
    echo ""
    echo -e "${GREEN}🎉 Demo complete!${NC}"
    echo "View the dashboard at: http://localhost:8000"
    echo "View API docs at: http://localhost:8000/docs"
}

# Parse commands
case "${1:-demo}" in
    demo)
        run_demo
        ;;
    fault)
        check_services
        trigger_fault "${2:-error_rate}" "${3:-60}"
        ;;
    status)
        check_services
        check_status
        ;;
    incidents)
        check_services
        list_incidents
        ;;
    incident)
        check_services
        get_incident "$2"
        ;;
    *)
        echo "Usage: $0 {demo|fault [type] [duration]|status|incidents|incident [id]}"
        echo ""
        echo "Commands:"
        echo "  demo              Run full demo sequence"
        echo "  fault [type]      Trigger a fault (error_rate, latency_spike, db_slow, memory_leak, dependency_down)"
        echo "  status            Check demo service status"
        echo "  incidents         List all incidents"
        echo "  incident [id]     Get incident details"
        exit 1
        ;;
esac

