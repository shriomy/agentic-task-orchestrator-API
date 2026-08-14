# Ticketmaster Discovery API — Calling Guide

This explains how to call the Ticketmaster Discovery API v2, and how it maps to `activities.py` / `search_events_tool` in this project.

## 1. Base setup

```dotenv
TICKETMASTER_API_KEY=your_key_here
TICKETMASTER_ROOT_URL=https://app.ticketmaster.com/discovery/v2/
```

> ⚠️ Never commit a real key to a repo or paste it into docs/chat. Treat it like a password — rotate it if it's ever been exposed, and load it only from environment variables (this project already does this via `config.py` / `settings.ticketmaster_api_key`).

**Authentication:** no headers, no OAuth — just an `apikey` query parameter on every request.

```
GET https://app.ticketmaster.com/discovery/v2/events.json?apikey=YOUR_KEY
```

## 2. General request shape

Every call to this API follows the same pattern:

```
GET {ROOT_URL}{resource}?{query_params}&apikey={API_KEY}
```

- **Method:** always `GET` (this is a read-only/search API — no POST/PUT/DELETE).
- **Format suffix:** `.json` on the resource path (e.g. `events.json`) — this is Ticketmaster's convention for explicitly requesting JSON, though the API defaults to JSON anyway.
- **Query params:** everything else (filters, paging, sorting) goes in the query string. There is no request body.
- **Response:** JSON, with results nested under an `_embedded` key (e.g. `_embedded.events`), plus a `page` object describing pagination (`size`, `totalElements`, `totalPages`, `number`).

## 3. The three endpoints this project cares about

| Endpoint | Purpose | Used by |
|---|---|---|
| `GET /discovery/v2/events` | Search events (concerts, sports, shows) | `search_events_tool` in `nodes.py` → `search_events()` in `activities.py` |
| `GET /discovery/v2/events/{id}` | Get one event's full details | not yet used — see §6 |
| `GET /discovery/v2/attractions` | Search "attractions" (artists, teams, venues-as-entities) — can substitute for POI/attraction search, similar role to `poi.py`'s OpenTripMap call | not yet used — see §6 |

### Events search — key parameters actually worth using

| Param | Type | Notes |
|---|---|---|
| `apikey` | string | required, every request |
| `keyword` | string | free-text search |
| `city` | array | comma-separated for multiple cities |
| `countryCode` | string | ISO country code, narrows results a lot |
| `startDateTime` / `endDateTime` | string, ISO-8601 (`2026-08-20T00:00:00Z`) | date window |
| `classificationName` | array | e.g. `music`, `sports` — filters by category |
| `size` | string/int | page size, default 20, max 200 |
| `page` | string/int | 0-indexed |
| `sort` | string | e.g. `date,asc` |

Everything else in the full parameter table (dmaId, promoterId, geoPoint, etc.) exists but is rarely needed for a general travel-events use case — only reach for them if a specific filter is requested.

## 4. Example calls

**Search events by city + date range:**
```
GET https://app.ticketmaster.com/discovery/v2/events.json
  ?apikey=YOUR_KEY
  &city=Rome
  &startDateTime=2026-09-01T00:00:00Z
  &endDateTime=2026-09-10T00:00:00Z
  &size=10
```

**Search events by keyword:**
```
GET https://app.ticketmaster.com/discovery/v2/events.json
  ?apikey=YOUR_KEY
  &keyword=jazz
  &city=Rome
```

**Get one event's details:**
```
GET https://app.ticketmaster.com/discovery/v2/events/G5vYZbZAdyCka.json
  ?apikey=YOUR_KEY
```

**Search attractions (artists/venues/teams):**
```
GET https://app.ticketmaster.com/discovery/v2/attractions.json
  ?apikey=YOUR_KEY
  &keyword=Coldplay
```

## 5. How this maps to the existing code

`activities.py` already implements the events search correctly:

```python
def search_events(city: str, keyword: str | None = None, start_date: str | None = None, end_date: str | None = None) -> dict:
    params = {
        "apikey": settings.ticketmaster_api_key,
        "city": city,
    }
    if keyword:
        params["keyword"] = keyword
    if start_date:
        params["startDateTime"] = start_date
    if end_date:
        params["endDateTime"] = end_date

    response = requests.get(
        "https://app.ticketmaster.com/discovery/v2/events.json",
        params=params,
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()
```

Notes on this implementation:
- `requests` builds the query string for you from the `params` dict — no need to hand-encode the URL.
- `response.raise_for_status()` throws an exception on 4xx/5xx (e.g. bad/missing key → 401, rate limit → 429) so errors don't silently return garbage. `TOOL_NODE` in `nodes.py` has `handle_tool_errors=True`, which catches this and turns it into a message the agent can see and react to.
- The base URL is currently hardcoded here rather than read from `settings.ticketmaster_root_url` — since you already have `TICKETMASTER_ROOT_URL` in your `.env`, it'd be more consistent to add that field to `config.py` and build the URL as `f"{settings.ticketmaster_root_url}events.json"`, so the root URL isn't duplicated in multiple files.

## 6. Possible extension: attractions as a second "POI-like" tool

Since `/discovery/v2/attractions` can serve a similar role to `poi.py`'s OpenTripMap call (searching things-to-see/do by keyword), you could add a `search_attractions()` function following the exact same pattern as `search_events()`, then expose it as a new `@tool` in `nodes.py` if you want the agent to be able to look up specific artists/venues/attractions instead of just general points of interest.

## 7. General checklist for calling any of these REST APIs safely

1. **Never hardcode keys** — always pull from `settings` (env vars), as this project already does.
2. **Always set a timeout** — this project uses `settings.request_timeout_seconds` everywhere; a hung request without a timeout can block your whole request pipeline.
3. **Always call `raise_for_status()`** (or equivalent) — don't assume a 200; let errors propagate to be handled explicitly.
4. **Only send parameters that have values** — build the `params` dict conditionally (as `activities.py` does) rather than sending `None`/empty values, which some APIs reject or misinterpret.
5. **Respect rate limits** — Ticketmaster's default developer key is capped (commonly 5 requests/sec, 5000/day); design retries/backoff if you expect volume.
6. **Paginate deliberately** — don't assume all results come back in one call; use `size`/`page` and check the `page.totalPages` field in the response if you need everything.


# OpenTripMap API — Calling Guide

This explains how to call the OpenTripMap API, and how it maps to `poi.py` / `search_places_tool` in this project.

## 1. Base setup

```dotenv
OPENTRIPMAP_API_KEY=your_key_here
```

> ⚠️ Rotate any key that's been pasted into chat/docs — treat it like a password. Load it only from environment variables, as this project already does via `config.py` / `settings.opentripmap_api_key`.

**Base URL pattern:**
```
http://api.opentripmap.com/0.1/{lang}/places/{endpoint}
```

- `{lang}` is a **path segment**, not a query param — e.g. `en` or `ru`.
- **Authentication:** `apikey` as a query parameter on every request, same pattern as Ticketmaster.
- **Method:** always `GET`, read-only.

## 2. The four endpoints

| Endpoint | Purpose | Returns |
|---|---|---|
| `GET /{lang}/places/geoname` | Turn a placename ("Rome") into coordinates | single object: `{name, country, lat, lon, population, timezone}` |
| `GET /{lang}/places/bbox` | List places inside a rectangular area (bounding box) | array of places (or GeoJSON) |
| `GET /{lang}/places/radius` | List places within N meters of a lat/lon point | array of places (or GeoJSON) — same shape as `bbox` |
| `GET /{lang}/places/xid/{xid}` | Full details for one specific place | single object with description, image, wiki links, etc. |

There's also autosuggest, not detailed in what you pasted, but the pattern is the same as the others.

### Important distinction (this affects your code — see §5)

- `geoname` **only** converts a name → coordinates. It does **not** accept `kinds`, `lat`, or `lon` as filters, and it does **not** return a list of places — it returns one geographic point.
- `bbox` and `radius` are the actual **"find me places to visit"** endpoints. They're the ones that take `kinds` (category), `lat`/`lon`, and return the list of POIs.

## 3. Key parameters by endpoint

**`geoname`**
| Param | Type | Required |
|---|---|---|
| `name` | string, placename | yes |
| `country` | string, 2-letter code | no |
| `apikey` | string | yes |

**`bbox`**
| Param | Type | Notes |
|---|---|---|
| `lon_min`, `lat_min`, `lon_max`, `lat_max` | number | required — defines the rectangle |
| `kinds` | string | comma-separated categories, e.g. `museums,churches` |
| `rate` | string | `1`–`3`, or `1h`–`3h` for cultural-heritage sites |
| `format` | string | `json`, `geojson` (default), or `count` |
| `limit` | integer | max results |
| `apikey` | string | required |

**`radius`** — same as `bbox`, but instead of a rectangle you give:
| Param | Type | Notes |
|---|---|---|
| `radius` | number | meters |
| `lon`, `lat` | number | center point |
| `kinds`, `rate`, `format`, `limit`, `apikey` | same as above | |

**`xid/{xid}`**
| Param | Type | Required |
|---|---|---|
| `xid` | string, path segment | yes — the object ID from a prior `bbox`/`radius` call |
| `apikey` | string | yes |

## 4. Example calls

**Step 1 — resolve a city name to coordinates:**
```
GET http://api.opentripmap.com/0.1/en/places/geoname
  ?name=Rome
  &apikey=YOUR_KEY
```
Response gives you `lat`/`lon` to use in the next call.

**Step 2 — find points of interest near that location (radius):**
```
GET http://api.opentripmap.com/0.1/en/places/radius
  ?lon=12.4964
  &lat=41.9028
  &radius=5000
  &kinds=museums,historic
  &format=json
  &limit=20
  &apikey=YOUR_KEY
```

**Or, find POIs in a bounding box:**
```
GET http://api.opentripmap.com/0.1/en/places/bbox
  ?lon_min=12.44
  &lat_min=41.87
  &lon_max=12.55
  &lat_max=41.93
  &kinds=churches
  &format=json
  &apikey=YOUR_KEY
```

**Step 3 — get full details for one result:**
```
GET http://api.opentripmap.com/0.1/en/places/xid/W286786280
  ?apikey=YOUR_KEY
```

This 3-step flow (name → coordinates → nearby places → details) is the normal way to use this API for a "show me things to do in X city" feature.

## 5. How this compares to the existing code — a bug worth fixing

Your current `poi.py`:

```python
def search_places(city: str | None = None, latitude: float | None = None, longitude: float | None = None, category: str | None = None) -> dict:
    params = {"apikey": settings.opentripmap_api_key}
    if city:
        params["city"] = city
    if latitude is not None:
        params["lat"] = latitude
    if longitude is not None:
        params["lon"] = longitude
    if category:
        params["kinds"] = category

    response = requests.get(
        "https://api.opentripmap.com/0.1/en/places/geoname",
        params=params,
        timeout=settings.request_timeout_seconds,
    )
```

This calls the **`geoname`** endpoint but passes it `lat`, `lon`, and `kinds` — parameters `geoname` doesn't recognize (it only accepts `name` and `country`). It's also passing `city` where the real param name is `name`. In practice, OpenTripMap will likely just ignore the unrecognized params and return a single geoname match — never an actual list of nearby places — which means `search_places_tool` in `nodes.py` currently can't do what its name implies (find POIs), it can only geocode a city name.

To actually search for places (which is presumably the intent, given the tool is meant to parallel Ticketmaster's event/attraction search), you'd want something closer to:

```python
def search_places(city: str, category: str | None = None, radius: int = 5000) -> dict:
    # Step 1: resolve city name -> coordinates
    geo_resp = requests.get(
        "https://api.opentripmap.com/0.1/en/places/geoname",
        params={"name": city, "apikey": settings.opentripmap_api_key},
        timeout=settings.request_timeout_seconds,
    )
    geo_resp.raise_for_status()
    geo = geo_resp.json()

    # Step 2: find POIs near those coordinates
    params = {
        "apikey": settings.opentripmap_api_key,
        "lon": geo["lon"],
        "lat": geo["lat"],
        "radius": radius,
        "format": "json",
    }
    if category:
        params["kinds"] = category

    places_resp = requests.get(
        "https://api.opentripmap.com/0.1/en/places/radius",
        params=params,
        timeout=settings.request_timeout_seconds,
    )
    places_resp.raise_for_status()
    return places_resp.json()
```

This makes two real API calls instead of one malformed one, but actually returns a usable list of POIs — matching what `search_places_tool`'s docstring/signature imply it should do.

## 6. General checklist (same as other API wrappers in this project)

1. **Never hardcode keys** — pull from `settings`.
2. **Always set a timeout** — `settings.request_timeout_seconds`, consistently.
3. **Always call `raise_for_status()`** — don't assume success.
4. **Only send parameters with values** — build `params` conditionally.
5. **Match parameters to the endpoint you're actually calling** — as shown above, sending `bbox`/`radius`-only params to `geoname` silently does nothing useful; always check which endpoint accepts which filters.
6. **Chain multi-step lookups explicitly** — geocode-then-search patterns (like this one) are common; don't try to force them into a single call.

# Booking.com (RapidAPI) — Calling Guide

This explains how to call the `booking-com15` RapidAPI, and how it maps to `hotels.py` / `search_accommodations_tool` in this project.

## 1. Base setup

```dotenv
HOTEL_SEARCH_API_KEY=your_rapidapi_key_here
```

> ⚠️ Rotate any key pasted into chat/docs. Load only from environment variables, as this project already does via `settings.hotel_search_api_key`.

**Authentication:** unlike Ticketmaster/OpenTripMap (query-param key), RapidAPI uses **headers**, not query params:

```
x-rapidapi-key: YOUR_KEY
x-rapidapi-host: booking-com15.p.rapidapi.com
```

Every request to any endpoint on this API needs both headers. `Content-Type: application/json` is also commonly sent, though it's not meaningful on a GET request with no body — harmless to include, not required.

**Method:** always `GET`, read-only.

## 2. This is a multi-step API — you can't search hotels in one call

This is the biggest difference from Ticketmaster/OpenTripMap. `searchHotels` does **not** take a plain city name — it takes a `dest_id` (a destination ID) and a `search_type` (e.g. `"CITY"`), both of which come from a *separate* destination-lookup endpoint first.

The realistic flow is:

```
1. searchDestination  (city name -> dest_id + search_type)
        ↓
2. searchHotels        (dest_id + search_type + dates -> list of hotels)
        ↓
3. getHotelDetails      (hotel_id + dates -> full details for one hotel)
```

(There's also `searchHotelsByCoordinates`, which skips step 1 if you already have lat/lon instead of a city name.)

## 3. Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/hotels/searchDestination` | Resolve a place name to a `dest_id` + `search_type` |
| `GET /api/v1/hotels/searchHotels` | Search hotels in a destination for given dates |
| `GET /api/v1/hotels/searchHotelsByCoordinates` | Same as above, but by lat/lon instead of `dest_id` |
| `GET /api/v1/hotels/getHotelDetails` | Full details for one specific hotel |
| `GET /api/v1/hotels/getSortBy` | Lookup table of valid `sort_by` values |
| `GET /api/v1/hotels/getFilter` | Lookup table of valid `categories_filter` values |
| `GET /api/v1/meta/getLanguages` | Lookup table of valid `languagecode` values |
| `GET /api/v1/meta/getCurrency` | Lookup table of valid `currency_code` values |

The `getSortBy`/`getFilter`/`getLanguages`/`getCurrency` endpoints exist because those parameters must be exact codes the API recognizes — you're expected to fetch and cache the valid values rather than guess them.

## 4. `searchHotels` — parameters

| Param | Required | Notes |
|---|---|---|
| `dest_id` | **yes** | from `searchDestination` |
| `search_type` | **yes** | from `searchDestination`, e.g. `CITY` |
| `arrival_date` | **yes** | `yyyy-mm-dd` |
| `departure_date` | **yes** | `yyyy-mm-dd` |
| `adults` | no | default 1 |
| `children_age` | no | comma-separated ages, e.g. `0,17` |
| `room_qty` | no | default 1 |
| `page_number` | no | default 1 |
| `price_min` / `price_max` | no | filters |
| `sort_by` | no | from `getSortBy` |
| `categories_filter` | no | from `getFilter` |
| `units` | no | `metric`/`imperial` |
| `temperature_unit` | no | `c`/`f` |
| `languagecode` | no | e.g. `en-us` |
| `currency_code` | no | e.g. `USD` |
| `location` | no | country bias, e.g. `US` |

## 5. Example calls

**Step 1 — resolve a city to a `dest_id`** (not shown in your paste, but this is the prerequisite call):
```
GET https://booking-com15.p.rapidapi.com/api/v1/hotels/searchDestination?query=Rome
Headers:
  x-rapidapi-key: YOUR_KEY
  x-rapidapi-host: booking-com15.p.rapidapi.com
```
Response includes a `dest_id` and `search_type` for each matching place — pick the right one (usually the top city-level match).

**Step 2 — search hotels:**
```
GET https://booking-com15.p.rapidapi.com/api/v1/hotels/searchHotels
  ?dest_id=-2092174
  &search_type=CITY
  &arrival_date=2026-09-01
  &departure_date=2026-09-05
  &adults=2
  &room_qty=1
  &page_number=1
  &currency_code=USD
Headers:
  x-rapidapi-key: YOUR_KEY
  x-rapidapi-host: booking-com15.p.rapidapi.com
```

**Step 3 — get details for one hotel from the results:**
```
GET https://booking-com15.p.rapidapi.com/api/v1/hotels/getHotelDetails
  ?hotel_id=191605
  &arrival_date=2026-09-01
  &departure_date=2026-09-05
  &adults=2
  &room_qty=1
  &currency_code=USD
Headers:
  x-rapidapi-key: YOUR_KEY
  x-rapidapi-host: booking-com15.p.rapidapi.com
```

### Response shape (from your example)

The hotel list is nested at `data.hotels[]`, and each entry's real fields live under `.property`, not at the top level:

```
data.hotels[i].property.name
data.hotels[i].property.id
data.hotels[i].property.reviewScore
data.hotels[i].property.priceBreakdown.grossPrice.value
data.hotels[i].property.priceBreakdown.grossPrice.currency
data.hotels[i].property.latitude / .longitude
data.hotels[i].property.photoUrls[0]
```

There's also an `appear[]` array in the response full of UI/tracking metadata for Booking.com's own app (banners, sign-in prompts, a giant tracking `contentUrl`) — this is noise you should discard/ignore when parsing; only `data.hotels` and `data.meta` (result count) matter for this use case.

## 6. How this compares to the existing code — real gaps to fix

Your current `hotels.py`:

```python
def search_accommodations(location: str, check_in: str, check_out: str, adults: int = 2, max_price: float | None = None) -> dict:
    params = {
        "location": location,
        "checkin_date": check_in,
        "checkout_date": check_out,
        "adults": adults,
    }
    if max_price is not None:
        params["max_price"] = max_price

    response = requests.get(
        "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchHotels",
        params=params,
        headers=headers,
        timeout=settings.request_timeout_seconds,
    )
```

This won't work as-is, for two reasons:

1. **Wrong parameter names.** The real API expects `dest_id`, `search_type`, `arrival_date`, `departure_date`, `price_min`/`price_max` — not `location`, `checkin_date`, `checkout_date`, `max_price`. As written, `searchHotels` will likely reject this call (missing required `dest_id`/`search_type`/`arrival_date`/`departure_date`) or silently ignore the unrecognized params.
2. **Missing the destination-lookup step.** There's no way to turn a plain city name like `"Rome"` into a `dest_id` without first calling `searchDestination`. A single-function `search_accommodations(location, ...)` can't satisfy this API's requirements — it needs to become two chained calls.

A corrected version:

```python
def search_accommodations(location: str, check_in: str, check_out: str, adults: int = 2, max_price: float | None = None) -> dict:
    if not settings.hotel_search_api_key:
        raise RuntimeError("HOTEL_SEARCH_API_KEY is required for accommodation search.")

    headers = {
        "x-rapidapi-key": settings.hotel_search_api_key,
        "x-rapidapi-host": "booking-com15.p.rapidapi.com",
    }

    # Step 1: resolve city name -> dest_id / search_type
    dest_resp = requests.get(
        "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchDestination",
        params={"query": location},
        headers=headers,
        timeout=settings.request_timeout_seconds,
    )
    dest_resp.raise_for_status()
    matches = dest_resp.json().get("data", [])
    if not matches:
        raise RuntimeError(f"No destination match found for '{location}'.")
    dest_id = matches[0]["dest_id"]
    search_type = matches[0]["search_type"]

    # Step 2: search hotels for that destination
    params = {
        "dest_id": dest_id,
        "search_type": search_type,
        "arrival_date": check_in,
        "departure_date": check_out,
        "adults": adults,
        "room_qty": 1,
        "currency_code": "USD",
    }
    if max_price is not None:
        params["price_max"] = max_price

    response = requests.get(
        "https://booking-com15.p.rapidapi.com/api/v1/hotels/searchHotels",
        params=params,
        headers=headers,
        timeout=settings.request_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()
```

(Field names in `searchDestination`'s response — `dest_id`, `search_type` — are illustrative here since it wasn't in your paste; worth confirming against a live call before relying on exact key names.)

## 7. General checklist (same pattern as the other API wrappers)

1. **Never hardcode keys** — pull from `settings`.
2. **Always set a timeout.**
3. **Always call `raise_for_status()`.**
4. **Only send parameters with values.**
5. **Auth style varies by provider** — Ticketmaster/OpenTripMap use a query-param `apikey`; RapidAPI providers like this one use headers (`x-rapidapi-key`, `x-rapidapi-host`). Check each API's docs rather than assuming one pattern fits all.
6. **Some APIs are multi-step by design** — don't assume a single "search" call can take free-text input; some require a resolve-then-search pattern like this one.
7. **Discard UI/tracking noise in responses** — this API in particular returns app-internal metadata (`appear[]`) alongside the real data; only parse the fields you actually need (`data.hotels`, `data.meta`).

