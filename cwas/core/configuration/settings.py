"""
Default category domain information for CWAS project.
"""
from copy import deepcopy

# This domain dictionary is incomplete.
# 'conservation' domains are added by BigWig file keys.
# 'gene_list' domains are added by gene matrix.
# 'region' domains are added by BED file keys.
_default_domains = {
    "variant_type": ["All", "SNV", "Indel"],
    "conservation": ["All"],
    "gene_list": ["Any"],
    "gencode": [  # GENCODE annotation categories
        "Any",
        "CodingRegion",
        "FrameshiftRegion",
        "InFrameRegion",
        "SilentRegion",
        "LoFRegion",
        "DamagingMissenseRegion",
        "MissenseRegion",
        "NoncodingRegion",
        "SpliceSiteNoncanonRegion",
        "IntronRegion",
        "PromoterRegion",
        "IntergenicRegion",
        "UTRsRegion",
        "lincRnaRegion",
        "OtherTranscriptRegion",
    ],
    "region": ["Any"],  # Custom annotation categories
}

_domain_types = list(_default_domains.keys())

# A category (domain combination) that includes a redundant domain pair will be
# excluded from CWAS analysis.
_redundant_domain_pairs = {
    ("variant_type", "gencode"): {
        ("All", "FrameshiftRegion"),
        ("All", "InFrameRegion"),
        ("All", "DamagingMissenseRegion"),
        ("All", "MissenseRegion"),
        ("All", "SilentRegion"),
        ("Indel", "DamagingMissenseRegion"),
        ("Indel", "MissenseRegion"),
        ("Indel", "SilentRegion"),
        ("SNV", "FrameshiftRegion"),
        ("SNV", "InFrameRegion"),
    },
    ("gene_list", "gencode"): {
        ("Any", "CodingRegion"),
        ("Any", "FrameshiftRegion"),
        ("Any", "InFrameRegion"),
        ("Any", "LoFRegion"),
        ("Any", "DamagingMissenseRegion"),
        ("Any", "MissenseRegion"),
        ("Any", "SilentRegion"),
        ("Any", "lincRnaRegion"),
        ("lincRNA", "Any")
    },
}

_AD_gene_sets = {f"Morabito2021.DEG.{direction}.{cell_type}" for direction in ["up", "down"] for cell_type in ["MG", "EX", "ASC", "ODC", "INH", "OPC"]}
_AD_regions = {'AD.MG.DARs', 'AD.EX.DARs', 'AD.ASC.DARs', 'AD.ODC.DARs', 'AD.MG.cCREs', 'AD.ODC.cCREs', 'AD.ASC.cCREs', 'AD.INH.cCREs', 'AD.EX.cCREs', 'AD.OPC.cCREs'}
_AD_redundant_domains = {(gene_set, region) for gene_set in _AD_gene_sets for region in _AD_regions if region.split('.')[1] != gene_set.split('.')[3]}
_redundant_domain_pairs[("gene_list", "region")] = _AD_redundant_domains

def get_default_domains() -> dict:
    return deepcopy(_default_domains)


def get_domain_types() -> list:
    return deepcopy(_domain_types)


def get_redundant_domain_pairs() -> dict:
    return deepcopy(_redundant_domain_pairs)
