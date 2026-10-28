import json
from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path('results/revision_1/dashboard');out.mkdir(parents=True,exist_ok=True)
report={'checks':[], 'console_errors':[], 'page_errors':[]}
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/opt/google/chrome/chrome', headless=True, args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1000})
    page.on('pageerror',lambda e: report['page_errors'].append(str(e)))
    page.on('console',lambda e: report['console_errors'].append(e.text) if e.type=='error' else None)
    page.goto('http://127.0.0.1:8765',wait_until='networkidle',timeout=60000)
    page.wait_for_function("document.getElementById('diagnosis-result-badge').textContent !== 'Memuat Model...'",timeout=60000)
    page.screenshot(path=str(out/'home.png'),full_page=True)
    for target in ['classification','regression','clustering']:
        page.locator('[data-target="section-'+target+'"]').first.click()
        page.wait_for_timeout(1300)
        page.screenshot(path=str(out/(target+'.png')),full_page=True)
        report['checks'].append({'section':target,'visible':page.locator('#section-'+target).is_visible()})
    report['selected_models']=page.evaluate("({classification: document.getElementById('model-select-class').value,regression:document.getElementById('model-select-reg').value})")
    report['result']=page.locator('#diagnosis-result-badge').inner_text()
    report['probability']=page.locator('#confidence-val').inner_text()
    report['z_scores']=page.evaluate("({HAZ:document.getElementById('val-z-haz').textContent,WAZ:document.getElementById('val-z-waz').textContent})")
    report['map_tile_images'] = page.locator('#spatial-map img.leaflet-tile').count()
    assert report['map_tile_images'] == 0
    report['province_options']=page.locator('#province-selector option').count()
    assert report['province_options'] == 34, report
    # Exercise rapid changes while keeping only the final inference on screen.
    page.locator('[data-target="section-classification"]').first.click()
    page.evaluate("""() => {
      const age = document.getElementById('slider-age-class');
      if (!age) return;
      for (const value of [3, 8, 12, 23]) { age.value = value; age.dispatchEvent(new Event('input', {bubbles: true})); }
    }""")
    page.wait_for_function("document.getElementById('diagnosis-result-badge').textContent !== 'Memuat Model...'",timeout=30000)
    page.locator('#btn-lang-en').click()
    page.wait_for_function("document.getElementById('diagnosis-result-badge').textContent !== 'Loading Model...'",timeout=30000)
    report['english_probability_label'] = page.locator('[data-lang-id="confidence_lbl"]').inner_text()
    assert 'uncalibrated' in report['english_probability_label']
    assert 'Overweight' not in page.locator('[data-lang-id="axis_waz_overweight"]').inner_text()
    parity=json.loads(Path('docs/models/revision_1/parity_samples.json').read_text())
    fixture=next(item for item in parity if item['task']=='classification' and item['code']=='et')
    features=fixture['features'][0]
    page.evaluate("""async f => {
        selectedGender = f.Gender_Encoded === 1 ? 'Laki-laki' : 'Perempuan';
        document.getElementById('slider-age-class').value=f['Age (months)'];
        document.getElementById('slider-height-class').value=f['Height (cm)'];
        document.getElementById('slider-weight-class').value=f['Weight (kg)'];
        syncForms('class');
        await runModelInference();
    }""",features)
    classes=['Normal','Severely Stunted','Stunted','Tall']
    expected=classes[int(fixture['prediction'][0])]
    assert page.locator('#diagnosis-result-badge').inner_text() == expected
    expected_probability=100*max(fixture['probabilities'][0])
    displayed=float(page.locator('#confidence-val').inner_text().strip('%'))
    assert abs(displayed-expected_probability) <= .0051,(displayed,expected_probability)
    report['ui_native_fixture']={'features':features,'class':expected,'probability_percent':displayed}
    report['age_limits'] = page.locator('input[type="range"]').evaluate_all("els => els.filter(e => e.id.includes('age')).map(e => ({id:e.id, min:e.min, max:e.max}))")
    assert all(item['max']=='24' for item in report['age_limits']), report
    assert not report['console_errors'], report['console_errors']
    page.set_viewport_size({'width':768,'height':1000})
    page.screenshot(path=str(out/'classification_768.png'),full_page=True)
    report['viewport_768'] = 'rendered'
    browser.close()
(out/'browser_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert not report['page_errors'],report['page_errors']
assert report['result']!='ERROR',report
