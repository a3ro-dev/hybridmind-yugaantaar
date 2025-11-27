"""Test script for HybridMind production features."""
import httpx
import json
import time

base_url = 'http://localhost:8000'

def test_bulk_edges():
    """Create edges between existing nodes."""
    print('=== Creating Edges ===')
    with httpx.Client() as client:
        resp = client.get(f'{base_url}/nodes?limit=10')
        nodes = resp.json()
        print(f'Found {len(nodes)} nodes')
        
        if len(nodes) >= 2:
            edges = []
            for i in range(len(nodes)-1):
                edges.append({
                    'source_id': nodes[i]['id'],
                    'target_id': nodes[i+1]['id'],
                    'type': 'related_to',
                    'weight': 0.8
                })
            
            resp = client.post(f'{base_url}/bulk/edges', json={'edges': edges, 'skip_validation': False})
            print('Edges created:', resp.json())

def test_search_caching():
    """Test hybrid search with caching."""
    search_data = {'query_text': 'attention mechanism in deep learning', 'top_k': 5}
    
    print('\n=== Testing Hybrid Search (First Call - Cache Miss) ===')
    with httpx.Client() as client:
        start = time.perf_counter()
        resp = client.post(f'{base_url}/search/hybrid', json=search_data)
        elapsed1 = (time.perf_counter() - start) * 1000
        result1 = resp.json()
        print(f'Query time (from API): {result1["query_time_ms"]}ms')
        print(f'Total request time: {elapsed1:.1f}ms')
        print(f'Results found: {len(result1["results"])}')
        for r in result1['results'][:3]:
            text_preview = r["text"][:60] + "..." if len(r["text"]) > 60 else r["text"]
            print(f'  - {text_preview} (score: {r["combined_score"]})')
    
    print('\n=== Testing Hybrid Search (Second Call - Should Be Cache Hit) ===')
    with httpx.Client() as client:
        start = time.perf_counter()
        resp = client.post(f'{base_url}/search/hybrid', json=search_data)
        elapsed2 = (time.perf_counter() - start) * 1000
        result2 = resp.json()
        print(f'Query time (from API): {result2["query_time_ms"]}ms')
        print(f'Total request time: {elapsed2:.1f}ms')
        print(f'Results found: {len(result2["results"])}')
    
    print(f'\n⚡ Cache speedup: {elapsed1/elapsed2:.1f}x faster!')
    
    print('\n=== Cache Stats After Searches ===')
    with httpx.Client() as client:
        resp = client.get(f'{base_url}/cache/stats')
        print(json.dumps(resp.json(), indent=2))

def test_rate_limiting():
    """Test rate limit headers."""
    print('\n=== Testing Rate Limit Headers ===')
    with httpx.Client() as client:
        resp = client.get(f'{base_url}/health')
        print(f'X-RateLimit-Limit: {resp.headers.get("X-RateLimit-Limit", "N/A")}')
        print(f'X-RateLimit-Remaining: {resp.headers.get("X-RateLimit-Remaining", "N/A")}')
        print(f'X-Process-Time-Ms: {resp.headers.get("X-Process-Time-Ms", "N/A")}')

def test_health_endpoints():
    """Test all health endpoints."""
    print('\n=== Testing Health Endpoints ===')
    with httpx.Client() as client:
        # Full health
        resp = client.get(f'{base_url}/health')
        health = resp.json()
        print(f'/health status: {health["status"]}')
        print(f'  - Uptime: {health["uptime_seconds"]}s')
        print(f'  - Embedding latency: {health["components"]["embedding"]["latency_ms"]}ms')
        print(f'  - CPU: {health["metrics"]["cpu_percent"]}%')
        print(f'  - Memory: {health["metrics"]["memory_percent"]}%')
        
        # Ready
        resp = client.get(f'{base_url}/ready')
        print(f'/ready: {resp.json()}')
        
        # Live
        resp = client.get(f'{base_url}/live')
        print(f'/live: {resp.json()}')

def test_rapid_cache():
    """Test cache with rapid repeated queries."""
    print('\n=== Rapid Search Test (10 identical queries) ===')
    search_data = {'query_text': 'neural network deep learning', 'top_k': 5}
    
    times = []
    with httpx.Client(base_url=base_url) as client:
        for i in range(10):
            start = time.perf_counter()
            resp = client.post('/search/hybrid', json=search_data)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            result = resp.json()
            api_time = result['query_time_ms']
            print(f'  Query {i+1}: {elapsed:.1f}ms total, {api_time}ms API time')
    
    print(f'\nFirst query: {times[0]:.1f}ms')
    print(f'Avg subsequent (cached): {sum(times[1:])/len(times[1:]):.1f}ms')
    print(f'Speedup from cache: {times[0]/times[-1]:.1f}x')
    
    # Check cache stats
    print('\n=== Final Cache Stats ===')
    with httpx.Client(base_url=base_url) as client:
        resp = client.get('/cache/stats')
        stats = resp.json()
        print(f'Hits: {stats["hits"]}')
        print(f'Misses: {stats["misses"]}')
        print(f'Hit Rate: {stats["hit_rate"]*100:.1f}%')


if __name__ == '__main__':
    print('=' * 60)
    print('HybridMind Production Features Test')
    print('=' * 60)
    
    test_bulk_edges()
    test_search_caching()
    test_rapid_cache()
    test_rate_limiting()
    test_health_endpoints()
    
    print('\n' + '=' * 60)
    print('✅ All tests completed!')
    print('=' * 60)

