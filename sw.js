const CACHE_NAME = 'kvcache-view-v7'
const urlsToCache = [
    './',
    './index.html',
    './hybrid-trends.html',
    './attention-trends.html',
    './context-demand.html',
    './cache-telemetry.html',
    './openrouter.css',
    './openrouter.js',
    './thumbnails/hybrid-trends.svg',
    './thumbnails/attention-trends.svg',
    './thumbnails/context-demand.svg',
    './thumbnails/cache-telemetry.svg',
    './inference.html',
    './train.html',
    './visualization.js',
    './cartridge-economics.html',
    './cartridge-economics.js',
    './cartridge-pricing.js',
    './cartridge-benchmarks.js',
    './thumbnails/cartridge-economics.png',
    './manifest.json',
    './icon-192.png',
    './icon-512.png',
]

self.addEventListener('install', (event) => {
    self.skipWaiting()
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll(urlsToCache)
        }),
    )
})

// Network-first for navigations/HTML so renamed or replaced pages don't get
// shadowed by a stale cache entry. Generated OpenRouter pages also need fresh
// shared styles/scripts when their charts change. Cache-first for other assets.
self.addEventListener('fetch', (event) => {
    const req = event.request
    const isHTML = req.mode === 'navigate' || (req.headers.get('accept') || '').includes('text/html')
    const url = new URL(req.url)
    const isOpenRouterAsset = url.origin === self.location.origin && /\/openrouter\.(css|js)$/.test(url.pathname)

    if (isHTML || isOpenRouterAsset) {
        event.respondWith(
            fetch(req)
                .then((response) => {
                    const copy = response.clone()
                    caches.open(CACHE_NAME).then((cache) => cache.put(req, copy))
                    return response
                })
                .catch(() =>
                    caches.match(req).then((r) => r || caches.match(isOpenRouterAsset ? url.pathname : './index.html')),
                ),
        )
        return
    }

    event.respondWith(
        caches.match(req).then((response) => {
            if (response) {
                return response
            }
            return fetch(req)
        }),
    )
})

self.addEventListener('activate', (event) => {
    const cacheWhitelist = [CACHE_NAME]
    event.waitUntil(
        caches
            .keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames.map((cacheName) => {
                        if (cacheWhitelist.indexOf(cacheName) === -1) {
                            return caches.delete(cacheName)
                        }
                    }),
                )
            })
            .then(() => self.clients.claim()),
    )
})
