"""
Builds text documents for the knowledge base.
Each document is a focused passage relevant to the 15 claims.
"""

import os

DOCUMENTS = {
    "intermittent_fasting.txt": """
Intermittent Fasting and Type 2 Diabetes

Intermittent fasting (IF) has been the subject of growing clinical interest for its potential metabolic benefits. Several studies published between 2018 and 2023 have examined its effects on glycaemic control in individuals with Type 2 diabetes.

A 2022 systematic review published in the journal Nutrients found that various forms of intermittent fasting, including time-restricted eating and alternate-day fasting, can significantly reduce HbA1c levels, fasting blood glucose, and body weight in people with Type 2 diabetes. The improvements were comparable to continuous caloric restriction.

A 2023 study from the University of Illinois Chicago, not Harvard, reported that time-restricted eating helped some patients achieve remission from Type 2 diabetes, meaning they maintained normal blood sugar levels without medication. The study involved 75 participants and found remission rates of around 50% over the study period—not 80%.

No 2023 Harvard study claiming to prove that intermittent fasting reverses Type 2 diabetes in 80% of patients has been identified in peer-reviewed literature. The claim appears to be a fabrication combining real research themes with false specifics.

The American Diabetes Association notes that diabetes remission is possible through weight loss interventions including dietary changes, but outcomes vary substantially by individual and long-term maintenance is challenging.
""",

    "mrna_vaccines_dna.txt": """
mRNA Vaccines and Human DNA: The Science

mRNA vaccines, such as those developed by Pfizer-BioNTech and Moderna for COVID-19, work by delivering messenger RNA instructions to cells to produce the SARS-CoV-2 spike protein, triggering an immune response.

A central concern raised by vaccine skeptics is whether mRNA can alter human DNA. This concern is scientifically unfounded for the following reasons:

1. Cellular compartmentalisation: mRNA is processed in the cytoplasm of cells, never entering the nucleus where DNA is housed. The mRNA in vaccines is coated in lipid nanoparticles that release it into the cytoplasm upon cell entry.

2. Lack of reverse transcriptase: Converting mRNA back into DNA (reverse transcription) requires the enzyme reverse transcriptase, which human cells do not normally produce. Only retroviruses like HIV carry their own reverse transcriptase.

3. mRNA degradation: Vaccine mRNA is inherently unstable and breaks down within days after injection. It does not persist in the body.

4. No mechanism for nuclear integration: Even if hypothetical reverse transcription occurred, DNA integration into chromosomes requires additional enzymes (integrases) not present in normal human cells.

These facts are confirmed by the CDC, WHO, the European Medicines Agency, and independent molecular biologists worldwide.
""",

    "brain_ten_percent_myth.txt": """
The 10% Brain Myth: Scientific Consensus

The idea that humans use only 10% of their brains is one of the most persistent neuroscience myths. Modern brain imaging technology has definitively disproven this claim.

Evidence against the 10% myth:

Functional MRI (fMRI) and PET scans show that over the course of a day, virtually all areas of the brain become active at some point. Even during sleep, many brain regions remain active.

From an evolutionary perspective, the brain consumes approximately 20% of the body's energy despite accounting for only 2% of body weight. Evolution would not maintain such a metabolically costly organ if 90% were unused.

Brain damage research: If 90% of the brain were dormant, damage to that 90% would cause no deficits. However, damage to virtually any brain region causes specific, measurable impairments.

Brain scans during complex tasks show widespread cortical activation, not isolated islands of activity.

Neurologists and neuroscientists universally classify the 10% claim as a myth. The origin is unclear—it may derive from misquoted William James writings or misinterpretation of glial cell research.
""",

    "exercise_cardiovascular_mental_health.txt": """
Exercise, Cardiovascular Health, and Mental Health

The health benefits of regular moderate-intensity aerobic exercise are among the most robustly established findings in medical science.

Cardiovascular Benefits:
The World Health Organization (WHO) recommends at least 150–300 minutes of moderate-intensity aerobic exercise per week for adults. A landmark 2018 meta-analysis published in the Lancet, analysing data from over 1.2 million individuals, found that exercise reduced the risk of cardiovascular disease by 35%.

The American Heart Association (AHA) notes that regular physical activity lowers blood pressure, reduces LDL cholesterol, helps maintain healthy weight, and reduces the risk of coronary artery disease, heart attack, and stroke.

A 2021 Cochrane review of 197 randomised controlled trials found that exercise interventions consistently reduced all-cause mortality and cardiovascular mortality.

Mental Health Benefits:
A 2018 study in Lancet Psychiatry analysed data from 1.2 million Americans and found that individuals who exercised regularly had 43% fewer days of poor mental health per month.

Exercise is recommended by NICE (UK), the APA (US), and WHO for the management of depression and anxiety. Meta-analyses show effect sizes for exercise on depression comparable to antidepressant medication in mild-to-moderate cases.

Mechanisms include increased BDNF (brain-derived neurotrophic factor), reduced cortisol, increased endorphins, and improved sleep quality.
""",

    "treaty_of_versailles.txt": """
Treaty of Versailles: Facts and Historical Context

The Treaty of Versailles was the primary peace settlement following World War I. Key facts:

Date of signing: The Treaty of Versailles was signed on 28 June 1919 in the Hall of Mirrors at the Palace of Versailles, France. It was not signed in 1918. The armistice that ended the fighting was signed on 11 November 1918, which is why that date is sometimes confused with the peace treaty.

Key provisions: The treaty held Germany responsible for the war (War Guilt Clause, Article 231), required Germany to pay reparations (eventually set at 132 billion gold marks), stripped Germany of territory, and severely restricted the German military.

The Great Depression: The Great Depression began in 1929, a full decade after the treaty. Historians identify multiple contributing causes:
- The US stock market crash of October 1929
- Bank failures across the United States and Europe
- The Smoot-Hawley Tariff Act (1930), which reduced international trade
- Overproduction in agriculture and industry during the 1920s
- Tight monetary policy by the Federal Reserve

While the Versailles reparations contributed to German economic instability and European financial fragility, historians do not identify the treaty as the direct or sole cause of the Great Depression. John Maynard Keynes famously criticised the treaty's economic terms in "The Economic Consequences of the Peace" (1919), but even he did not predict the Depression.

The causal chain from Versailles to the Great Depression is indirect and contested.
""",

    "berlin_wall_german_reunification.txt": """
The Fall of the Berlin Wall and German Reunification

The Berlin Wall:
The Berlin Wall fell on the night of 9 November 1989. East German authorities announced that citizens could cross the border freely, and crowds immediately began dismantling the wall. This event is widely regarded as the symbolic end of the Cold War, although the Cold War is also dated to the dissolution of the Soviet Union on 26 December 1991.

The wall had divided East and West Berlin since 13 August 1961.

German Reunification:
German reunification occurred on 3 October 1990—not 9 November 1990. This date is now a public holiday in Germany, known as German Unity Day (Tag der Deutschen Einheit).

The reunification process:
- March 1990: First free elections in East Germany
- July 1990: Economic and monetary union
- 3 October 1990: Political reunification — East Germany (the German Democratic Republic) officially ceased to exist and its states joined the Federal Republic of Germany

The Two Plus Four Treaty, signed on 12 September 1990, provided the international framework for reunification by the four WWII Allied powers (USA, USSR, UK, France) plus East and West Germany.
""",

    "china_population_gdp.txt": """
China's Population and Economic Status (2023–2024)

Population:
For decades, China was the world's most populous country. However, according to the United Nations Population Fund (UNFPA) and Indian government data, India surpassed China in population in 2023, becoming the world's most populous country with approximately 1.429 billion people versus China's 1.412 billion.

This demographic shift is a significant milestone. China's population began declining due to the long-term effects of the one-child policy (1980–2015), aging demographics, and falling birth rates.

GDP Rankings:
As of 2023:
- Nominal GDP: The United States remains the world's largest economy at approximately $27.4 trillion. China is second at approximately $17.7 trillion. 
- GDP by Purchasing Power Parity (PPP): China is considered the largest economy by some measures of PPP, depending on the methodology used by the IMF or World Bank.

The claim that China has the world's largest economy by nominal GDP is incorrect; it is the second largest.
""",

    "nelson_mandela.txt": """
Nelson Mandela: Imprisonment and Presidency

Nelson Mandela was a South African anti-apartheid activist and leader of the African National Congress (ANC).

Imprisonment:
Mandela was convicted of sabotage and conspiracy to overthrow the government at the Rivonia Trial in 1964 and sentenced to life imprisonment. He served 27 years in prison before his release on 11 February 1990.

However, the 27 years were not all spent on Robben Island:
- Robben Island: 1964–1982 (approximately 18 years)
- Pollsmoor Prison, Cape Town: 1982–1988
- Victor Verster Prison, Paarl: 1988–1990

The claim that he was imprisoned on Robben Island for 27 years overstates the time spent at that facility; the full 27-year figure refers to his total imprisonment across three locations.

Presidency:
Following his release, Mandela led the ANC in negotiations to end apartheid. In the South African general election of 27 April 1994—the first election in which all races could vote—the ANC won a majority. Nelson Mandela was inaugurated as President of South Africa on 10 May 1994, becoming the country's first Black and first democratically elected president.

He served one term (1994–1999) and did not seek re-election.
""",

    "global_gdp_growth.txt": """
Global GDP Growth Rates and Economic History

Global GDP growth refers to the year-over-year increase in the total economic output of all countries combined.

2023 GDP Growth:
According to the International Monetary Fund (IMF) World Economic Outlook (April 2024), global GDP growth in 2023 was 3.1%. The World Bank's estimate was similar at 2.6%. Neither organisation reported anything near 15% growth.

Historical Context:
The fastest periods of global economic growth in recorded history include:
- Post-World War II reconstruction (1950s–1960s): Global growth rates of 4–5% annually in some years
- Post-2020 pandemic recovery (2021): Global GDP rebounded by approximately 6.0% after contracting 3.1% in 2020
- The 1960s saw high growth rates in developed economies

A 15% global GDP growth rate has never been recorded in modern economic history. Such a figure would represent the combined output growth of all nations simultaneously expanding at extraordinary rates, which has no historical precedent.

The claim of 15% growth in 2023 overstates actual performance by approximately five times.
""",

    "income_inequality_g7.txt": """
Income Inequality in G7 Nations: Gini Coefficient Analysis

The Gini coefficient is the standard measure of income inequality, ranging from 0 (perfect equality) to 1 (total inequality). Higher values indicate greater inequality.

Gini Coefficients for G7 nations (OECD data, most recent available, approximately 2021–2022):
- United States: 0.394–0.414 (among developed nations, consistently highest)
- United Kingdom: 0.357
- Italy: 0.347
- Canada: 0.307
- Japan: 0.334
- France: 0.292
- Germany: 0.296

The United States consistently ranks as the most unequal G7 country by the Gini coefficient. This is attributed to lower union coverage, more regressive tax structures, higher CEO-to-worker pay ratios, and weaker social safety nets compared to European G7 members.

The World Bank and OECD both confirm the US has the highest income inequality among G7 nations on Gini measures, though some measures of wealth inequality (as opposed to income inequality) show even starker gaps.
""",

    "us_unemployment_2008.txt": """
US Unemployment: Historical Data and the 2008 Financial Crisis

The 2008 Financial Crisis and Unemployment:
The 2008 global financial crisis triggered by the collapse of the US housing market led to a significant rise in unemployment. According to the US Bureau of Labor Statistics (BLS):

- January 2008: US unemployment rate was 5.0%
- Peak unemployment: 10.0% in October 2009
- Recovery: Unemployment gradually declined through the 2010s, reaching pre-crisis levels around 2015–2016

The unemployment rate never reached 25% during or after the 2008 crisis. The 10% peak was the highest since the early 1980s recession.

The Great Depression Comparison:
The 25% unemployment figure refers to the Great Depression. US unemployment peaked at approximately 24.9% in 1933, the worst point of the Depression. Some estimates place it even higher when accounting for underemployment and workers who had given up seeking employment.

The 2008–2009 recession is sometimes called the "Great Recession" but was substantially less severe than the Great Depression in terms of unemployment magnitude.
""",

    "billionaire_wealth_covid_vaccines.txt": """
Billionaire Wealth During COVID-19 and Vaccine Development

Billionaire Wealth Growth:
Oxfam International reported in 2022 that the world's billionaires saw their wealth approximately double during the first two years of the pandemic (March 2020 to November 2021). Specifically, the wealth of the world's ten richest men doubled during the pandemic. Forbes data showed the global billionaire population's combined wealth grew from roughly $8 trillion to over $13.8 trillion in two years.

This wealth concentration occurred while the International Labour Organization estimated that pandemic-related job losses pushed tens of millions into poverty.

COVID-19 Vaccine Development:
COVID-19 vaccines were developed through a combination of public and private investment:
- Operation Warp Speed (US): $18 billion in public funding
- BARDA (US) and equivalent European agencies provided billions in development and manufacturing guarantees
- NIH researchers co-invented the mRNA stabilisation technology used in Pfizer-BioNTech and Moderna vaccines

The vaccines were developed at unprecedented speed not primarily due to profit incentives but due to emergency authorisation frameworks, pre-funding of manufacturing before trial completion, international coordination, and existing mRNA research.

Conflating billionaire wealth gains—driven primarily by tech and financial sector stock appreciation—with pharmaceutical vaccine development conflates two distinct phenomena. While pharmaceutical executives did profit, the claim that vaccines were developed "primarily to profit" ignores the enormous public investment and the genuine public health emergency context.
""",

    "gpt4_bar_exam.txt": """
GPT-4 Performance on the Bar Exam

The Uniform Bar Exam (UBE) is a standardised bar examination used in many US states. Passing typically requires a score of 266–270 out of 400, depending on jurisdiction.

GPT-4 Bar Exam Performance:
OpenAI's technical report for GPT-4 (released March 2023) reported that GPT-4 scored approximately 298–310 on the UBE, placing it around the 90th percentile of human test-takers—meaning it scored better than approximately 90% of people who take the exam.

Several independent researchers and law school clinics subsequently tested GPT-4 on bar exam practice questions and reached similar conclusions, validating OpenAI's reported scores.

This is in contrast to its predecessor GPT-3.5, which scored around the 10th percentile on the same exam.

The 90th percentile corresponds to the top 10% of human test-takers, which aligns with the claim.

Some researchers have noted methodological considerations, such as whether GPT-4 may have seen bar exam questions in training data. However, as of available evaluations, independent verification largely supports the top 10% claim.
""",

    "ai_job_replacement.txt": """
AI and the Future of Employment

Claims that artificial intelligence will replace all jobs or cause universal unemployment are not supported by mainstream economic research.

Major Research Findings:

McKinsey Global Institute (2023): Estimated that AI could automate 25–46% of work activities by 2030–2045, but distinguished between automating tasks within jobs versus eliminating entire jobs. Most scenarios project job transformation, not elimination.

World Economic Forum (Future of Jobs Report 2023): Estimated that 26% of current jobs could be disrupted by technology by 2027, but that 69 million new jobs would be created while 83 million existing roles disappear—a net displacement of 14 million jobs, not 100% unemployment.

OECD (2023): Approximately 14% of jobs in OECD countries are at high risk of automation, with another 32% likely to change significantly. Historical precedent (agricultural revolution, industrial revolution) shows technology transforms labour markets without permanent mass unemployment.

Oxford Study (Frey & Osborne, 2013): The widely-cited study that estimated 47% of US jobs were at risk was a worst-case scenario; subsequent research significantly revised these estimates downward.

Economic consensus: Economists broadly agree that AI will change the nature of work substantially, may cause transitional unemployment in affected sectors, and may exacerbate inequality—but predictions of 100% unemployment are not found in credible academic or institutional research.
""",

    "iphone_history.txt": """
The Original iPhone: Launch Features and Timeline

Apple introduced the first iPhone at Macworld Conference & Expo on 9 January 2007. It went on sale to the public on 29 June 2007 in the United States.

Original iPhone Features (2007):
- 3.5-inch multi-touch display
- 2-megapixel camera (no front-facing camera)
- Mobile Safari browser
- iPod music player
- Visual Voicemail
- Google Maps integration
- Wi-Fi and Bluetooth
- EDGE (2G) network connectivity — NOT 3G

Features NOT present at original launch:
- App Store: Apple launched the App Store on 10 July 2008, alongside iPhone OS 2.0. Third-party applications were not available at the original launch.
- 3G connectivity: The original iPhone used EDGE (2.5G) for cellular data. The iPhone 3G, released on 11 July 2008, was the first iPhone with 3G connectivity.
- Copy and paste
- MMS messaging

Steve Jobs presented the original iPhone as "an iPod, a phone, and an internet communicator" combined into one device, emphasising the revolutionary touch interface.

The claim that the original iPhone included an App Store and 3G connectivity at launch is incorrect on both counts.
"""
}


def build_knowledge_base(kb_dir: str = "data/knowledge_base") -> None:
    """Write all documents to the knowledge base directory."""
    os.makedirs(kb_dir, exist_ok=True)
    for filename, content in DOCUMENTS.items():
        path = os.path.join(kb_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip())
    print(f"[KB] Wrote {len(DOCUMENTS)} documents to '{kb_dir}'")


if __name__ == "__main__":
    build_knowledge_base()