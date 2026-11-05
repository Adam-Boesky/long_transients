"""Follow-up target directories (`Source_Analysis.followup`) and their readers."""
from __future__ import annotations

import pytest
import yaml

from Source_Analysis import followup
from source_db.identity import coord_slug
from source_db.lightcurves import custom

RA, DEC = 352.5509585158785, -24.804052230076703


@pytest.fixture
def root(tmp_path):
    t = followup.create_target(RA, DEC, root=tmp_path, list_id=6, redshift=0.57, notes='',
                               spectra=[{'file': 'spectra/LDSS3_20250626.txt',
                                         'plot': 'spectra/LDSS3_20250626.pdf'}])
    (t.path / followup.PHOTOMETRY_FILE).write_text(
        '# MJD Mag Mag_Err Filter UL Telescope Instrument\n'
        '60825.4 19.74 0.06 r False Magellan LDSS3c\n')
    (t.path / 'zfps').mkdir()
    (t.path / 'zfps' / 'batchfp_req0000000001_lc.txt').write_text('')
    return tmp_path


def test_directory_is_the_website_slug(root):
    t, = followup.iter_targets(root)
    assert t.slug == coord_slug(RA, DEC) == '352p550959_n24p804052'
    assert t.path == root / t.slug
    assert t.redshift == 0.57 and t.meta['list_id'] == 6
    assert t.spectra[0]['file'] == t.path / 'spectra' / 'LDSS3_20250626.txt'
    assert [p.name for p in t.zfps_files] == ['batchfp_req0000000001_lc.txt']


def test_yaml_keeps_its_key_order(root):
    text = (root / coord_slug(RA, DEC) / followup.TARGET_FILE).read_text()
    assert list(yaml.safe_load(text))[:6] == ['slug', 'ra', 'dec', 'list_id', 'redshift', 'notes']


def test_a_position_that_does_not_match_the_directory_is_refused(root):
    path = root / coord_slug(RA, DEC) / followup.TARGET_FILE
    meta = yaml.safe_load(path.read_text())
    meta['dec'] = -25.804052230076703          # the kind of typo #6's raw file name has
    path.write_text(yaml.safe_dump(meta, sort_keys=False))
    with pytest.raises(ValueError, match='not the directory name'):
        followup.iter_targets(root)


def test_create_never_overwrites(root):
    with pytest.raises(FileExistsError):
        followup.create_target(RA, DEC, root=root)


def test_nearest_target_within_radius(root):
    assert followup.nearest_target(RA + 1e-4, DEC, root=root).slug == coord_slug(RA, DEC)
    assert followup.nearest_target(RA + 1e-2, DEC, root=root) is None


def test_load_targets_summarises(root):
    df = followup.load_targets(root)
    assert df.loc[0, ['n_spectra', 'has_photometry', 'n_zfps']].tolist() == [1, True, 1]


def test_custom_photometry_reads_the_target(root):
    files = custom.file_positions(root)
    assert files['custom_id'].tolist() == [coord_slug(RA, DEC)]
    phot = custom.read_photometry(files.loc[0, 'path'], files.loc[0, 'custom_id'])
    assert phot[['band', 'telescope']].iloc[0].tolist() == ['r', 'Magellan']
