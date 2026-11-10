#!/usr/bin/env python3
"""Compare rain-only and wet/overcast runs without altering training images."""

import argparse
import json
from pathlib import Path

from PIL import Image

from run import fit_tile, read_rgb, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--after', required=True, type=Path)
    args = parser.parse_args()
    before, after = args.before.resolve(), args.after.resolve()
    baseline = json.loads((before / 'manifest.json').read_text())
    updated = json.loads((after / 'manifest.json').read_text())
    previous = {(r['source'], r['weather']): r for r in baseline['records'] if r['variant'] == 0}
    current = {(r['source'], r['weather']): r for r in updated['records'] if r['variant'] == 0}
    sources = sorted({r['source'] for r in updated['records']})
    destination = after / 'previews'
    labels = ['Original', 'Rain only / medium', 'Wet + overcast / medium', 'Wet + overcast / heavy']
    overview = Image.new('RGB', (640 * 4, 512 * len(sources)), '#eeeeee')
    scenes = json.loads((after / 'config_snapshot/scenes.json').read_text())
    for row, name in enumerate(sources):
        old = previous[name, 'rain_medium']
        medium = current[name, 'rain_overcast_wet_medium']
        heavy = current[name, 'rain_overcast_wet_heavy']
        if old['source_sha256'] != medium['source_sha256']:
            raise ValueError(f'Source differs between runs: {name}')
        source = Path(updated['input_dir']) / name
        if sha256(source) != medium['source_sha256']:
            raise ValueError(f'Source has changed: {source}')
        paths = [source, before / old['output'], after / medium['output'], after / heavy['output']]
        sheet = Image.new('RGB', (2560, 512), '#eeeeee')
        contacts = Image.new('RGB', (3200, 342), '#eeeeee')
        for col, (label, path) in enumerate(zip(labels, paths)):
            image = read_rgb(path)
            tile = fit_tile(image, 640, 480, scenes[name]['label'] + ' / ' + label)
            sheet.paste(tile, (col * 640, 0))
            overview.paste(tile, (col * 640, row * 512))
            width, height = image.size
            contact = image.crop((int(width * 0.12), int(height * 0.67), int(width * 0.89), int(height * 0.92)))
            contacts.paste(fit_tile(contact, 800, 310, label), (col * 800, 0))
        stem = Path(name).stem
        sheet.save(destination / (stem + '_before_after.jpg'), quality=95)
        contacts.save(destination / (stem + '_contact_comparison.jpg'), quality=95)
    overview.save(destination / 'before_after_overview.jpg', quality=94)
    print(f'Saved before/after and wheel-contact comparisons for {len(sources)} sources: {destination}')


if __name__ == '__main__':
    main()
