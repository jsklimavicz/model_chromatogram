from chromatogram.peakcreator import PeakCreator

peak_creator = PeakCreator()

peaks = [
    peak_creator.peak(retention_time=1.006, height=12, name="toluene"),
    peak_creator.peak(retention_time=1.801, height=25, name="benzene"),
    peak_creator.peak(retention_time=2.498, height=103, name="4-butylphenol"),
    peak_creator.peak(retention_time=5.141, height=87, name="thymol"),
    peak_creator.peak(retention_time=5.569, height=80, name="carvacrol"),
    peak_creator.peak(
        retention_time=6.067, height=20, name="2-isopropylphenol", base_width=0.4
    ),
    peak_creator.peak(retention_time=9.163, height=24, name="phenethyl alcohol"),
    peak_creator.peak(retention_time=13.22, height=11, name="1-phenylethanol"),
    # add some junk to the baseline so it don't look too clean
    peak_creator.peak(
        retention_time=1.101, height=0.5, base_width=0.6, base_asymmetry=1.2
    ),
    peak_creator.peak(
        retention_time=2.934, height=1.1, base_width=0.5, base_asymmetry=1.3
    ),
    peak_creator.peak(
        retention_time=3.843, height=0.8, base_width=0.8, base_asymmetry=1.2
    ),
    peak_creator.peak(
        retention_time=5.924, height=1.2, base_width=0.45, base_asymmetry=2.5
    ),
    peak_creator.peak(
        retention_time=6.542, height=2.5, base_width=0.6, base_asymmetry=1.3
    ),
    peak_creator.peak(
        retention_time=9.11, height=2.4, base_width=0.7, base_asymmetry=1.6
    ),
    peak_creator.peak(
        retention_time=12.38, height=0.7, base_width=0.8, base_asymmetry=0.2
    ),
]
