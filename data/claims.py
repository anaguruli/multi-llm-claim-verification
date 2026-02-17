"""
15 claims with ground-truth verdicts, categories, and justifications.
Mix of TRUE / FALSE / PARTIALLY_TRUE / MISLEADING across 4 categories.
"""

CLAIMS = [
    # ── Scientific & Health ──────────────────────────────────────────────────
    {
        "id": "c01",
        "text": (
            "A 2023 Harvard study proved that intermittent fasting reverses "
            "Type 2 diabetes in 80% of patients."
        ),
        "category": "scientific_health",
        "ground_truth": "FALSE",
        "justification": (
            "No 2023 Harvard study with those exact findings exists. "
            "Research shows intermittent fasting can improve glycaemic control "
            "and sometimes achieve remission, but the '80%' figure and 'reverses' "
            "framing are unsupported fabrications."
        ),
    },
    {
        "id": "c02",
        "text": (
            "COVID-19 mRNA vaccines alter the human DNA because mRNA is "
            "converted back into DNA inside the nucleus."
        ),
        "category": "scientific_health",
        "ground_truth": "FALSE",
        "justification": (
            "mRNA never enters the cell nucleus and cannot be reverse-transcribed "
            "into DNA by normal human cellular machinery. The claim contradicts "
            "established molecular biology."
        ),
    },
    {
        "id": "c03",
        "text": (
            "The human brain uses only 10% of its capacity at any given time, "
            "leaving 90% dormant and untapped."
        ),
        "category": "scientific_health",
        "ground_truth": "FALSE",
        "justification": (
            "Brain-imaging studies show virtually all regions are active at some "
            "point; the 10% myth has been thoroughly debunked by neuroscience."
        ),
    },
    {
        "id": "c04",
        "text": (
            "Regular moderate-intensity aerobic exercise has been shown to reduce "
            "the risk of cardiovascular disease and improve mental health outcomes."
        ),
        "category": "scientific_health",
        "ground_truth": "TRUE",
        "justification": (
            "Extensive peer-reviewed evidence from organisations such as the WHO, "
            "AHA, and numerous meta-analyses confirms both cardiovascular and "
            "mental-health benefits of regular aerobic exercise."
        ),
    },
    # ── Historical & Geopolitical ─────────────────────────────────────────────
    {
        "id": "c05",
        "text": (
            "The Treaty of Versailles was signed in 1918 and directly caused "
            "the Great Depression."
        ),
        "category": "historical_geopolitical",
        "ground_truth": "FALSE",
        "justification": (
            "The Treaty was signed on 28 June 1919, not 1918. While the harsh "
            "reparations contributed to European economic instability, historians "
            "cite multiple causes of the Great Depression (1929 stock crash, bank "
            "failures, Smoot-Hawley tariffs). 'Directly caused' is an "
            "oversimplification."
        ),
    },
    {
        "id": "c06",
        "text": (
            "The Berlin Wall fell on 9 November 1989, marking the symbolic end "
            "of the Cold War and German reunification occurred exactly one year "
            "later on 9 November 1990."
        ),
        "category": "historical_geopolitical",
        "ground_truth": "PARTIALLY_TRUE",
        "justification": (
            "The Wall fell on 9 November 1989 — TRUE. German reunification "
            "occurred on 3 October 1990, not 9 November 1990 — FALSE. "
            "The claim is partially true."
        ),
    },
    {
        "id": "c07",
        "text": (
            "China is the world's most populous country and also has the world's "
            "largest economy by nominal GDP."
        ),
        "category": "historical_geopolitical",
        "ground_truth": "PARTIALLY_TRUE",
        "justification": (
            "India surpassed China as the world's most populous country in 2023. "
            "The United States has the largest nominal GDP; China is second "
            "(though first by PPP). Both sub-claims are false as of 2024."
        ),
    },
    {
        "id": "c08",
        "text": (
            "Nelson Mandela was imprisoned on Robben Island for 27 years before "
            "becoming South Africa's first democratically elected president in 1994."
        ),
        "category": "historical_geopolitical",
        "ground_truth": "PARTIALLY_TRUE",
        "justification": (
            "Mandela served 27 years in prison total, but only 18 of those were "
            "on Robben Island; he was later transferred to Pollsmoor and Victor "
            "Verster prisons. He did become South Africa's first democratically "
            "elected president in 1994. Partially true."
        ),
    },
    # ── Statistical & Economic ────────────────────────────────────────────────
    {
        "id": "c09",
        "text": (
            "Global GDP grew by 15% in 2023, making it the fastest growth year "
            "in recorded history."
        ),
        "category": "statistical_economic",
        "ground_truth": "FALSE",
        "justification": (
            "Global GDP growth in 2023 was approximately 3.1% according to IMF "
            "data. The claim inflates the figure by roughly 5×; post-WWII "
            "reconstruction and early-pandemic-recovery years saw higher rates."
        ),
    },
    {
        "id": "c10",
        "text": (
            "The United States has the highest income inequality among all G7 "
            "nations as measured by the Gini coefficient."
        ),
        "category": "statistical_economic",
        "ground_truth": "TRUE",
        "justification": (
            "OECD and World Bank Gini data consistently show the US has the "
            "highest income inequality among G7 members, with a Gini around "
            "0.39–0.41 versus lower values for Germany, France, Japan, etc."
        ),
    },
    {
        "id": "c11",
        "text": (
            "Unemployment in the United States reached a record high of 25% "
            "during the 2008 financial crisis."
        ),
        "category": "statistical_economic",
        "ground_truth": "FALSE",
        "justification": (
            "US unemployment peaked at about 10% in October 2009 after the 2008 "
            "crisis. The 25% figure corresponds to the Great Depression peak "
            "(1933), not 2008."
        ),
    },
    {
        "id": "c12",
        "text": (
            "Billionaires' collective wealth doubled during the COVID-19 pandemic "
            "while millions fell into poverty, suggesting vaccines were developed "
            "primarily to profit pharmaceutical companies."
        ),
        "category": "statistical_economic",
        "ground_truth": "MISLEADING",
        "justification": (
            "The wealth-doubling statistic is approximately accurate (Oxfam data). "
            "However, conflating wealth accumulation with vaccine profit motives "
            "is misleading; vaccines were developed with substantial public funding "
            "and in unprecedented timeframes driven by emergency need, not "
            "primarily profit."
        ),
    },
    # ── Technology & AI ───────────────────────────────────────────────────────
    {
        "id": "c13",
        "text": (
            "GPT-4 has been independently verified to pass the bar exam with a "
            "score in the top 10% of human test-takers."
        ),
        "category": "technology_ai",
        "ground_truth": "TRUE",
        "justification": (
            "OpenAI's technical report and subsequent independent evaluations "
            "confirmed GPT-4 scored around the 90th percentile on the Uniform "
            "Bar Exam — top 10% of human test-takers."
        ),
    },
    {
        "id": "c14",
        "text": (
            "Artificial intelligence will replace all human jobs within the next "
            "10 years, leading to 100% unemployment globally."
        ),
        "category": "technology_ai",
        "ground_truth": "FALSE",
        "justification": (
            "No credible economic or AI research supports 100% unemployment within "
            "10 years. Most studies (McKinsey, WEF, OECD) project partial "
            "automation of tasks with net job transformation, not total elimination."
        ),
    },
    {
        "id": "c15",
        "text": (
            "The first iPhone was released in 2007 and introduced features like "
            "an App Store and 3G connectivity at launch."
        ),
        "category": "technology_ai",
        "ground_truth": "PARTIALLY_TRUE",
        "justification": (
            "The original iPhone launched on 29 June 2007 — TRUE. However, it "
            "did not include an App Store (added with iPhone OS 2.0 in July 2008) "
            "nor 3G (added with iPhone 3G in 2008). Partially true."
        ),
    },
]

VERDICT_LABELS = ["TRUE", "FALSE", "PARTIALLY_TRUE", "MISLEADING"]
CATEGORIES = ["scientific_health", "historical_geopolitical", "statistical_economic", "technology_ai"]