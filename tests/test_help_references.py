"""PSD help topics keep citing the literature their methods follow."""

from __future__ import annotations

import pytest

from nlab.help.content import load_topics

CITED = {
    "psd": ("10.1016/0029-554X(59)90067-9", "10.1016/0168-9002(92)90617-D"),
    "mca-charge": ("10.1016/0029-554X(59)90067-9", "10.1109/NSSMIC.2014.7431222"),
    "mca-zerocross": ("10.1016/0029-554X(61)90198-7", "10.1016/j.nima.2010.06.252"),
    "fom": ("10.1016/0029-554X(71)90054-1", "10.1016/0167-5087(84)90016-4"),
    "psd-energy-fom": ("10.1088/1748-0221/15/03/P03030", "10.1016/j.nima.2020.164080"),
    "waveform": ("10.1016/j.nima.2008.06.004", "10.1002/9781119410225"),
    "waveform-optimization": ("10.1016/j.nima.2023.168184", "10.1016/j.nima.2023.168486"),
}


@pytest.mark.parametrize("topic", sorted(CITED))
def test_psd_method_topics_cite_their_references(topic: str) -> None:
    markdown = load_topics()[topic].markdown
    assert "**References**" in markdown
    for doi in CITED[topic]:
        assert f"doi:{doi}" in markdown
