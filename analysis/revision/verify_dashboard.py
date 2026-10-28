"""Check the static dashboard in Chromium without changing research outputs."""

import functools
import argparse
import csv
import http.server
import json
import threading
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'scratch/dashboard_validation'


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def run(live_url=None):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'docs/revision_manifest.json').read_text())
    fixtures = json.loads((ROOT / 'docs' / manifest['model_directory'] / 'parity_samples.json').read_text())
    report = {'checks': [], 'page_errors': [], 'console_errors': [], 'expected_network_errors': [],
              'model_requests': [], 'tile_requests': [], 'accessibility': []}
    with (ROOT / 'results/clustering_revised/province_profiles.csv').open() as stream:
        regional = {row['Province']: row for row in csv.DictReader(stream)}
    handler = functools.partial(QuietHandler, directory=str(ROOT / 'docs'))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            chrome = Path('/opt/google/chrome/chrome')
            launch = {'headless': True, 'args': ['--no-sandbox']}
            if chrome.exists():
                launch['executable_path'] = str(chrome)
            browser = p.chromium.launch(**launch)
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            fault = {'active': False}
            def record_console(message):
                if message.type == 'error':
                    key = 'expected_network_errors' if fault['active'] and 'net::ERR_FAILED' in message.text else 'console_errors'
                    report[key].append(message.text)
            page.on('console', record_console)
            page.on('pageerror', lambda error: report['page_errors'].append(error.stack))
            page.on('request', lambda request: report['model_requests'].append(request.url)
                    if '/models/' in request.url else None)
            page.on('request', lambda request: report['tile_requests'].append(request.url)
                    if any(host in request.url for host in ['basemaps.cartocdn', 'tile.openstreetmap', '/tiles/']) else None)
            url = live_url or f'http://127.0.0.1:{server.server_port}'
            page.goto(url, wait_until='domcontentloaded', timeout=60000)
            targets = page.locator('#site-navigation .nav-item-btn').evaluate_all('els=>els.map(el=>el.dataset.target)')
            assert targets == ['section-home', 'section-clustering', 'section-classification', 'section-regression'], targets
            assert page.locator('#section-methods').count() == 0
            report['checks'].append('Navigation contains home and the three analysis tasks only')
            page.wait_for_function('revisionManifest !== null && mapObject !== null', timeout=60000)
            page.screenshot(path=str(OUTPUT / 'home_1440.png'))
            page.set_viewport_size({'width': 390, 'height': 844})
            page.screenshot(path=str(OUTPUT / 'home_390.png'))
            assert not report['model_requests'], 'Home eagerly downloads model JSON'
            report['checks'].append('Home downloads no model JSON')
            page.evaluate('''() => {
                document.querySelector('#site-navigation [data-target="section-clustering"]').click();
                document.querySelector('#site-navigation [data-target="section-home"]').click();
            }''')

            def navigate(section):
                button = page.locator(f'.nav-item-btn[data-target="section-{section}"]').first
                if not button.is_visible():
                    page.locator('#mobile-menu-toggle').click()
                button.click()
                page.wait_for_function('(id) => !document.getElementById(id).classList.contains("hidden")',
                                       arg=f'section-{section}')

            navigate('clustering')
            page.wait_for_function('document.querySelectorAll("#spatial-map path.leaflet-interactive").length > 0')
            assert not report['model_requests'], 'The map downloads supervised model JSON'
            report['checks'].append('Map downloads no model JSON')

            for width in [320, 390, 768, 1024, 1440]:
                page.set_viewport_size({'width': width, 'height': 900})
                for section in ['home', 'classification', 'regression', 'clustering']:
                    navigate(section)
                    dimensions = page.locator('#main-content').evaluate('el => ({width:el.clientWidth, scroll:el.scrollWidth})')
                    assert dimensions['scroll'] <= dimensions['width'] + 1, (width, section, dimensions)
                    assert dimensions['width'] >= min(width - 48, 270), (width, section, dimensions)
                report['checks'].append(f'All four sections reflow at {width}px')
            page.set_viewport_size({'width': 390, 'height': 844})
            page.locator('#mobile-menu-toggle').click()
            assert page.locator('#mobile-menu-toggle').get_attribute('aria-expanded') == 'true'
            page.keyboard.press('Shift+Tab')
            assert page.locator('#btn-lang-en').evaluate('el => el === document.activeElement')
            page.keyboard.press('Tab')
            assert page.locator('#mobile-menu-close').evaluate('el => el === document.activeElement')
            page.keyboard.press('Escape')
            assert page.locator('#mobile-menu-toggle').get_attribute('aria-expanded') == 'false'
            assert page.locator('#mobile-menu-toggle').evaluate('el => el === document.activeElement')
            report['checks'].append('Mobile menu closes with Escape and restores focus')

            page.set_viewport_size({'width': 1440, 'height': 1000})
            page.locator('#btn-lang-en').click()
            assert page.locator('html').get_attribute('lang') == 'en'
            for task, section, select in [('classification', 'classification', 'model-select-class'), ('HAZ', 'regression', 'model-select-reg')]:
                navigate(section)
                fixture = next(item for item in fixtures if item['task'] == task)
                features = fixture['features'][-10]
                prefix = 'class' if task == 'classification' else 'reg'
                page.locator(f'#btn-gender-female-{prefix}').click()
                for variable, key in [('age', 'Age (months)'), ('height', 'Height (cm)'), ('weight', 'Weight (kg)')]:
                    page.locator(f'#slider-{variable}-{prefix}').evaluate(
                        '(el, value) => {el.value=value;el.dispatchEvent(new Event("input",{bubbles:true}));}', features[key])
                for code in manifest['available'][task.lower()]:
                    page.select_option(f'#{select}', code)
                    page.wait_for_function('([section,code]) => {const el=document.getElementById(section);return el.dataset.state==="ready" && el.dataset.model===code;}',
                                           arg=[f'section-{section}', code], timeout=90000)
                    for target in ['classification'] if task == 'classification' else ['HAZ', 'WAZ']:
                        reference = next(item for item in fixtures if item['task'] == target and item['code'] == code)
                        if target == 'classification':
                            expected = manifest['classes'][int(reference['prediction'][-10])]
                            assert page.locator('#diagnosis-result-badge').inner_text() == expected, code
                            probability = float(page.locator('#confidence-val').inner_text().strip('%'))
                            assert abs(probability - 100 * max(reference['probabilities'][-10])) <= .0051
                        else:
                            value = float(page.locator(f'#val-z-{target.lower()}').inner_text().replace(' SD', ''))
                            assert abs(value - reference['prediction'][-10]) <= .0051, (target, code, value)
                    report['checks'].append(f'UI native parity: {task}/{code}')

            for age in [0, 23, 24]:
                page.locator('#slider-age-reg').evaluate('(el,v)=>{el.value=v;el.dispatchEvent(new Event("input",{bubbles:true}));}', age)
                page.wait_for_function('document.getElementById("section-regression").dataset.state === "ready"')
                assert page.locator('#slider-age-reg').get_attribute('aria-valuenow') is None
                assert str(age) in page.locator('#slider-age-reg').get_attribute('aria-valuetext')
            page.locator('#btn-gender-male-reg').click()
            page.locator('#btn-gender-male-reg').focus()
            page.keyboard.press('ArrowRight')
            assert page.locator('#btn-gender-female-reg').get_attribute('aria-checked') == 'true'

            # A failed classification download must leave regression usable and support retry.
            navigate('classification')
            page.evaluate('delete window.stuntingPredictor.models.classification_rf')
            route = '**/models/paper_final/rf.json'
            fault['active'] = True
            page.route(route, lambda request: request.abort())
            page.select_option('#model-select-class', 'rf')
            page.wait_for_function('document.getElementById("section-classification").dataset.state === "error"')
            assert page.locator('#confidence-val').inner_text() == '—'
            assert not page.locator('#diagnostic-action-box').is_visible()
            navigate('regression')
            page.wait_for_function('document.getElementById("section-regression").dataset.state === "ready"')
            assert page.locator('#val-z-haz').inner_text() != '—'
            navigate('classification')
            page.wait_for_function('document.getElementById("section-classification").dataset.state === "error"')
            page.unroute(route)
            page.locator('#retry-classification').click()
            page.wait_for_function('document.getElementById("section-classification").dataset.state === "ready"', timeout=60000)
            fault['active'] = False
            report['checks'].append('Failed classification download clears stale output, leaves regression usable, and supports retry')

            navigate('regression')
            page.wait_for_function('document.getElementById("section-regression").dataset.state === "ready"')
            for value in [3, 8, 12, 23]:
                page.locator('#slider-age-reg').evaluate('(el,v)=>{el.value=v;el.dispatchEvent(new Event("input",{bubbles:true}));}', value)
            page.wait_for_function('document.getElementById("section-regression").dataset.state === "ready"')
            report['checks'].append('Rapid input changes settle on the latest input')
            session = page.context.new_cdp_session(page)
            session.send('Network.enable')
            session.send('Network.setCacheDisabled', {'cacheDisabled': True})
            session.send('Network.emulateNetworkConditions', {'offline': False, 'latency': 250,
                         'downloadThroughput': 10000000, 'uploadThroughput': 10000000})
            page.evaluate('delete window.stuntingPredictor.models.haz_linear; delete window.stuntingPredictor.models.waz_linear')
            page.select_option('#model-select-reg', 'linear')
            state = page.locator('#section-regression').get_attribute('data-state')
            assert state == 'loading', state
            page.wait_for_function('document.getElementById("section-regression").dataset.state === "ready"')
            session.send('Network.emulateNetworkConditions', {'offline': False, 'latency': 0,
                         'downloadThroughput': -1, 'uploadThroughput': -1})
            session.send('Network.setCacheDisabled', {'cacheDisabled': False})
            page.evaluate('''() => {
                const select=document.getElementById('model-select-reg');
                for(const value of ['linear','ridge','rf']) {select.value=value;select.dispatchEvent(new Event('change',{bubbles:true}));}
            }''')
            page.wait_for_function('document.getElementById("section-regression").dataset.state === "ready" && document.getElementById("section-regression").dataset.model === "rf"')
            report['checks'].append('Slow downloads show loading; rapid model switches retain the final selection')
            page.locator('#btn-gender-male-reg').click()
            page.locator('#slider-age-reg').evaluate('(el)=>{el.value=0;el.dispatchEvent(new Event("input",{bubbles:true}));}')
            page.wait_for_function('document.getElementById("section-regression").dataset.state === "ready"')
            for target in ['HAZ', 'WAZ']:
                reference = next(item for item in fixtures if item['task'] == target and item['code'] == 'rf')
                value = float(page.locator(f'#val-z-{target.lower()}').inner_text().replace(' SD', ''))
                assert abs(value - reference['prediction'][-5]) <= .0051
            report['checks'].append('Male and female UI encoding agrees with native fixtures')

            navigate('clustering')
            page.wait_for_function('!document.getElementById("section-clustering").classList.contains("hidden") && document.querySelectorAll("#spatial-map path.leaflet-interactive").length > 0')
            assert page.evaluate('Number.isFinite(mapObject.getCenter().lat) && Number.isFinite(mapObject.getCenter().lng)')
            provinces = page.locator('#province-selector option').evaluate_all('els => els.map(el=>el.value)')
            assert len(provinces) == 34
            for province in provinces:
                page.select_option('#province-selector', province)
                expected = [float(regional[province][key]) for key in ['Stunting_2019', 'KEK', 'Immunization', 'BBLR']]
                actual = [page.locator(f'#prov-{key}-val').inner_text() for key in ['stunting', 'kek', 'imunisasi', 'bblr']]
                # JS toFixed and Python formatting differ at half-cent ties.
                assert all(abs(float(value.strip('%'))-raw) <= .0051 for value, raw in zip(actual, expected)), (province, actual, expected)
                for name, column in [('kmeans','Group_KMeans'),('gmm','Group_GMM'),('hc','Group_Hierarchical')]:
                    assert int(page.locator(f'#{name}-cluster-num').text_content()) == int(regional[province][column]), (province, name)
            for metric in page.locator('#map-metric-selector option').evaluate_all('els=>els.map(e=>e.value)'):
                page.select_option('#map-metric-selector', metric)
            for algorithm in ['Klaster_KMeans', 'Klaster_GMM', 'Klaster_HC']:
                page.select_option('#map-cluster-selector', algorithm)
                assert page.locator('#cluster-profile-grid').inner_text().strip()
            for province in ['Kalimantan Utara', 'Kepulauan Riau', 'Sulawesi Barat']:
                page.select_option('#province-selector', province)
                assert 'unavailable' in page.locator('#province-geometry-note').inner_text().lower(), province
            page.locator('#reset-map').click()
            assert page.locator('#spatial-map img.leaflet-tile').count() == 0
            assert not report['tile_requests']
            report['checks'].append('34 province profiles, four indicators, three algorithms, missing geometry, and tile-free map')

            for language in ['id', 'en']:
                page.locator(f'#btn-lang-{language}').click()
                assert page.locator('html').get_attribute('lang') == language
            report['checks'].append('Both languages render on the analysis pages')

            axe = OUTPUT / 'axe-4.10.3.min.js'
            if not axe.exists():
                urllib.request.urlretrieve('https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.3/axe.min.js', axe)
            page.add_script_tag(path=str(axe))
            for width in [1440, 390]:
                page.set_viewport_size({'width': width, 'height': 1000 if width == 1440 else 844})
                for section in ['home', 'classification', 'regression', 'clustering']:
                    navigate(section)
                    if section == 'classification':
                        assert 'Screening' not in page.locator('[data-lang-id="diag_header"]').inner_text()
                        assert page.locator('.probability-heading').evaluate('''el => {
                            const [label, value] = [...el.children].map(child => child.getBoundingClientRect());
                            return label.right + 7 <= value.left || label.bottom <= value.top;
                        }'''), 'Probability label overlaps its value'
                    page.screenshot(path=str(OUTPUT / f'{section}_{width}.png'))
                    result = page.evaluate('''async () => {
                        const r=await axe.run(document,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21a','wcag21aa','wcag22aa']}});
                        return {violations:r.violations.map(v=>({id:v.id,impact:v.impact,nodes:v.nodes.map(n=>({target:n.target,summary:n.failureSummary}))})),incomplete:r.incomplete.map(v=>({id:v.id,targets:v.nodes.map(n=>n.target)}))};
                    }''')
                    report['accessibility'].append({'width': width, 'section': section, **result})
            session = page.context.new_cdp_session(page)
            tree = session.send('Accessibility.getFullAXTree')
            report['chrome_accessibility_nodes'] = len(tree['nodes'])
            report['screen_reader'] = 'Chrome accessibility tree and keyboard checked; physical screen reader not tested'
            assert not report['page_errors'], report['page_errors']
            assert not report['console_errors'], report['console_errors']
            violations = [item for item in report['accessibility'] if item['violations']]
            assert not violations, json.dumps(violations)
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        (OUTPUT / 'browser_smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {len(report["checks"])} browser checks; {len(report["accessibility"])} axe audits')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', help='Check a deployed dashboard instead of the local server')
    args = parser.parse_args()
    if args.url:
        OUTPUT = OUTPUT / 'live'
    run(args.url)
