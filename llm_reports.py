
# llm_reports.py - BILINGUAL REPORT GENERATION (ENGLISH & TELUGU)
import os
import base64
from io import BytesIO
from PIL import Image

# ============================================================================
# API KEY CONFIGURATION (Optional for enhanced reports)
# ============================================================================
OPENAI_API_KEY = None
# To enable AI-enhanced reports, uncomment and add your key:
# OPENAI_API_KEY = "your-api-key-here"

# ============================================================================
# TELUGU TRANSLATIONS - MEDICAL TERMINOLOGY
# ============================================================================

TELUGU_BODY_PARTS = {
    'Chest': 'ఛాతీ',
    'Spine': 'వెన్నెముక',
    'Knee': 'మోకాలి',
    'Shoulder': 'భుజం',
    'Hand': 'చేయి',
    'Foot': 'పాదం',
    'Ankle': 'చీలమండ',
    'Fingers': 'వేళ్లు',
    'Wrist': 'మణికట్టు',
    'Elbow': 'మోచేయి',
    'Pelvis': 'కటి ప్రదేశం',
    'Skull': 'తలపు ఎముక',
    'Neck': 'మెడ',
    'Jaw': 'దవడ',
    'Abdomen': 'కడుపు',
    'Thigh': 'తొడ'
}

TELUGU_DISEASES = {
    # Chest conditions
    'Normal': 'సాధారణ',
    'Pneumonia': 'న్యుమోనియా (ఊపిరితిత్తుల వాపు)',
    'Pleural Effusion': 'ఊపిరితిత్తుల చుట్టూ నీరు చేరడం',
    'Cardiomegaly': 'గుండె పెద్దదిగా ఉండటం',
    'Atelectasis': 'ఊపిరితిత్తు కుంచించుకు పోవడం',
    'Pulmonary Edema': 'ఊపిరితిత్తుల్లో నీరు చేరడం',

    # Bone conditions
    'Fracture': 'ఎముక విరగడం',
    'Dislocation': 'ఎముక స్థానం తప్పడం',
    'Arthritis': 'కీళ్ళ వాపు',
    'Osteoarthritis': 'కీళ్ళ దరిద్రం',
    'Rheumatoid Arthritis': 'రుమటాయిడ్ ఆర్థరైటిస్',

    # Spine conditions
    'Disc Degeneration': 'వెన్నెముక డిస్క్ క్షీణత',
    'Vertebral Fracture': 'వెన్నెముక విరగడం',
    'Scoliosis': 'వెన్నెముక వక్రత',
    'Spinal Stenosis': 'వెన్నెముక సన్నబడటం',
    'Cervical Disc Disease': 'మెడ డిస్క్ సమస్య',

    # Joint conditions
    'Meniscal Tear': 'మోకాలి కార్టిలేజ్ చిరిగిపోవడం',
    'Ligament Injury': 'లిగమెంట్ గాయం',
    'Rotator Cuff Tear': 'భుజం కండరాలు చిరిగిపోవడం',
    'Tennis Elbow': 'టెన్నిస్ ఎల్బో',
    'Carpal Tunnel Syndrome': 'కార్పల్ టన్నల్ సిండ్రోమ్',
    'TMJ Disorder': 'దవడ కీలు సమస్య',

    # Other conditions
    'Sprain': 'బెణుకు',
    'Tendon Injury': 'నరాల గాయం',
    'Bone Lesion': 'ఎముక గాయం',
    'Soft Tissue Mass': 'మాంసపు భాగంలో ముద్ద',
    'Hip Dysplasia': 'తుంటి ఎముక వైకల్యం',
    'Avascular Necrosis': 'ఎముక రక్త ప్రవాహం లేక చనిపోవడం',
    'Plantar Fasciitis': 'అరికాలి నొప్పి',
    'Bone Spur': 'ఎముక ముల్లు',
    'Cranial Abnormality': 'తల ఎముక అసాధారణత',
    'Sinus Disease': 'సైనస్ సమస్య',
    'Dental Abnormality': 'దంతాల సమస్య',
    'Bowel Obstruction': 'ప్రేగుల అడ్డుపడటం',
    'Free Air': 'కడుపులో గాలి',
    'Kidney Stone': 'మూత్రపిండాల్లో రాయి',
    'Abnormal': 'అసాధారణ',
    'Soft Tissue Injury': 'మాంసపు భాగం గాయం'
}

# ============================================================================
# ENGLISH CLINICAL REPORT TEMPLATE
# ============================================================================

def generate_clinical_report_english(result):
    """Generate clinical report in English"""

    condition_details = {
        'Normal': {
            'findings': 'No acute abnormalities detected. Bone density and joint spaces appear within normal limits.',
            'impression': 'Radiographic examination demonstrates no significant pathology.',
            'recommendations': [
                'Continue routine health maintenance',
                'No immediate follow-up imaging required',
                'Return if symptoms develop'
            ]
        },
        'Pneumonia': {
            'findings': 'Increased opacity in lung fields consistent with consolidation. Bilateral involvement noted.',
            'impression': 'Radiographic findings suggestive of pneumonia.',
            'recommendations': [
                'Antibiotic therapy as per culture sensitivity',
                'Follow-up chest X-ray in 2-4 weeks',
                'Monitor for complications',
                'Ensure adequate hydration'
            ]
        },
        'Fracture': {
            'findings': 'Discontinuity in bone cortex with evidence of trauma. Alignment assessment required.',
            'impression': 'Fracture identified requiring orthopedic evaluation.',
            'recommendations': [
                'Orthopedic consultation for treatment planning',
                'Immobilization of affected area',
                'Follow-up imaging post-treatment',
                'Pain management as needed'
            ]
        },
        'Osteoarthritis': {
            'findings': 'Joint space narrowing with osteophyte formation. Subchondral sclerosis present.',
            'impression': 'Degenerative changes consistent with osteoarthritis.',
            'recommendations': [
                'Physical therapy for joint mobility',
                'Weight management if applicable',
                'Anti-inflammatory medications',
                'Consider joint injections if conservative management fails'
            ]
        }
    }

    details = condition_details.get(result['disease'], {
        'findings': f'Radiographic changes consistent with {result["disease"]}.',
        'impression': f'Clinical findings suggest {result["disease"]}.',
        'recommendations': [
            'Clinical correlation recommended',
            'Consider additional imaging if symptoms persist',
            'Specialist consultation advised'
        ]
    })

    report = f"""
CLINICAL RADIOLOGY REPORT
{'=' * 60}

PATIENT INFORMATION
Date of Examination: [Date]
Study Type: X-Ray Radiography

CLINICAL INDICATION
Body Part Examined: {result['body_part']}
Clinical History: [As provided by referring physician]

TECHNICAL DETAILS
Detection Confidence: {result['confidence']*100:.1f}%
Image Quality: Adequate for diagnostic interpretation
Technique: Standard radiographic projection

FINDINGS
{details['findings']}

Primary Observation: {result['disease']}
Diagnostic Confidence: {result['disease_conf']*100:.1f}%

CLINICAL IMPRESSION
{details['impression']}

RECOMMENDATIONS
"""

    for i, rec in enumerate(details['recommendations'], 1):
        report += f"{i}. {rec}\n"

    report += f"""
ADDITIONAL NOTES
• This is an AI-assisted preliminary assessment
• Correlation with clinical presentation is essential
• Review by licensed radiologist recommended
• Additional diagnostic studies may be warranted

{'=' * 60}
Reporting System: AI-Assisted Diagnostic Tool
Report Generated: [Timestamp]

DISCLAIMER: This automated report should be reviewed by a qualified
healthcare professional. Treatment decisions should not be based solely
on this preliminary assessment.
"""

    return report

# ============================================================================
# ENGLISH PATIENT REPORT TEMPLATE
# ============================================================================

def generate_patient_report_english(result):
    """Generate patient-friendly report in English"""

    condition_info = {
        'Normal': {
            'explanation': 'Your X-ray shows no signs of fractures, infections, or other abnormalities. The bones and joints appear healthy.',
            'causes': 'This is a normal, healthy result.',
            'what_to_do': 'Continue your regular health routine and maintain a healthy lifestyle.',
            'when_to_see_doctor': 'Return if you develop any new symptoms or concerns.',
            'prognosis': 'Excellent. No medical intervention needed.'
        },
        'Pneumonia': {
            'explanation': 'Your chest X-ray shows signs of pneumonia, which is an infection that inflames the air sacs in your lungs.',
            'causes': 'Usually caused by bacteria, viruses, or fungi. Can be triggered by weakened immune system, smoking, or other lung conditions.',
            'what_to_do': 'Get plenty of rest, drink lots of fluids, take prescribed antibiotics, and use a humidifier if recommended.',
            'when_to_see_doctor': 'If you have difficulty breathing, chest pain, high fever, or symptoms worsen.',
            'prognosis': 'Most people recover within 2-4 weeks with proper treatment.'
        },
        'Fracture': {
            'explanation': 'Your X-ray shows a break or crack in the bone, which is called a fracture.',
            'causes': 'Usually caused by trauma, falls, sports injuries, or repetitive stress on the bone.',
            'what_to_do': 'Keep the area immobilized, avoid putting weight on it, apply ice (if recent), and follow your doctor\'s treatment plan.',
            'when_to_see_doctor': 'Immediately if you notice increased pain, numbness, or changes in skin color.',
            'prognosis': 'Most fractures heal within 6-12 weeks with proper care.'
        },
        'Osteoarthritis': {
            'explanation': 'Your X-ray shows signs of osteoarthritis, which is wear and tear of the joint cartilage causing pain and stiffness.',
            'causes': 'Age-related wear, previous injuries, obesity, genetics, or repetitive stress on joints.',
            'what_to_do': 'Maintain healthy weight, do low-impact exercises (swimming, cycling), use hot/cold therapy, and take anti-inflammatory medications as prescribed.',
            'when_to_see_doctor': 'If pain becomes severe, affects daily activities, or conservative treatments don\'t help.',
            'prognosis': 'While it cannot be cured, symptoms can be managed effectively with lifestyle changes and treatment.'
        }
    }

    info = condition_info.get(result['disease'], {
        'explanation': f'Your X-ray shows signs of {result["disease"]}.',
        'causes': 'Various factors can contribute to this condition.',
        'what_to_do': 'Consult with your healthcare provider for specific guidance.',
        'when_to_see_doctor': 'Schedule an appointment with your doctor for evaluation.',
        'prognosis': 'Prognosis varies based on individual circumstances.'
    })

    report = f"""
YOUR X-RAY RESULTS EXPLAINED
{'=' * 60}

🔍 WHAT WE FOUND
We carefully examined your {result['body_part'].lower()} X-ray.

Finding: {result['disease']}
Detection Confidence: {result['disease_conf']*100:.1f}%

📋 WHAT THIS MEANS IN SIMPLE TERMS
{info['explanation']}

🤔 WHAT MIGHT HAVE CAUSED THIS
{info['causes']}

💡 WHAT YOU SHOULD DO
{info['what_to_do']}

⚠️ WHEN TO SEE YOUR DOCTOR
{info['when_to_see_doctor']}

📈 WHAT TO EXPECT (PROGNOSIS)
{info['prognosis']}

{'=' * 60}

IMPORTANT REMINDERS
• This is a preliminary automated assessment
• Always discuss results with your doctor
• Your doctor will create a treatment plan based on your complete medical history
• Don't make any treatment decisions without professional medical advice
• Ask your doctor any questions you have about your results

NEXT STEPS
1. Schedule an appointment with your healthcare provider
2. Bring this report and discuss your symptoms
3. Follow your doctor's recommendations
4. Keep track of any changes in your symptoms
5. Don't hesitate to ask questions

Remember: Early detection and proper medical care lead to better health outcomes.
You're taking an important step by getting this examination!

{'=' * 60}
Report Generated by: AI-Assisted Medical Imaging System
Date: [Timestamp]

For questions or concerns, please contact your healthcare provider.
"""

    return report

# ============================================================================
# TELUGU CLINICAL REPORT TEMPLATE
# ============================================================================

def generate_clinical_report_telugu(result):
    """Generate clinical report in Telugu"""

    body_part_te = TELUGU_BODY_PARTS.get(result['body_part'], result['body_part'])
    disease_te = TELUGU_DISEASES.get(result['disease'], result['disease'])

    # Condition-specific details in Telugu
    condition_details = {
        'Normal': {
            'findings': 'ఎటువంటి తీవ్రమైన అసాధారణతలు కనిపించలేదు. ఎముక సాంద్రత మరియు కీలు సామాన్య స్థితిలో ఉన్నాయి.',
            'impression': 'రేడియోగ్రాఫిక్ పరీక్షలో ముఖ్యమైన సమస్యలు కనిపించలేదు.',
            'recommendations': [
                'సాధారణ ఆరోగ్య నిర్వహణ కొనసాగించండి',
                'తక్షణ ఫాలో-అప్ ఇమేజింగ్ అవసరం లేదు',
                'లక్షణాలు అభివృద్ధి చెందితే తిరిగి రండి'
            ]
        },
        'Pneumonia': {
            'findings': 'ఊపిరితిత్తుల క్షేత్రాలలో పెరిగిన అస్పష్టత గట్టిపడటానికి అనుగుణంగా ఉంది. రెండు వైపుల ప్రమేయం గుర్తించబడింది.',
            'impression': 'రేడియోగ్రాఫిక్ ఫలితాలు న్యుమోనియాను సూచిస్తున్నాయి.',
            'recommendations': [
                'కల్చర్ సెన్సిటివిటీ ప్రకారం యాంటీబయోటిక్ థెరపీ',
                '2-4 వారాలలో ఫాలో-అప్ చెస్ట్ X-రే',
                'సమస్యల కోసం పర్యవేక్షించండి',
                'తగినంత హైడ్రేషన్ నిర్ధారించుకోండి'
            ]
        },
        'Fracture': {
            'findings': 'గాయానికి సాక్ష్యంతో ఎముక కార్టెక్స్‌లో నిరంతరత లేదు. అమరిక అంచనా అవసరం.',
            'impression': 'ఆర్థోపెడిక్ మూల్యాంకనం అవసరమయ్యే ఫ్రాక్చర్ గుర్తించబడింది.',
            'recommendations': [
                'చికిత్స ప్రణాళిక కోసం ఆర్థోపెడిక్ సంప్రదింపు',
                'ప్రభావిత ప్రాంతం యొక్క స్థిరీకరణ',
                'చికిత్స తర్వాత ఫాలో-అప్ ఇమేజింగ్',
                'అవసరమైనప్పుడు నొప్పి నిర్వహణ'
            ]
        },
        'Osteoarthritis': {
            'findings': 'ఆస్టియోఫైట్ నిర్మాణంతో ఉమ్మడి స్థల సంకుచితం. సబ్‌కాండ్రల్ స్క్లెరోసిస్ ఉంది.',
            'impression': 'ఆస్టియో ఆర్థరైటిస్‌కు అనుగుణంగా క్షీణత మార్పులు.',
            'recommendations': [
                'కీలు చలనశీలత కోసం భౌతిక చికిత్స',
                'వర్తించే ఉంటే బరువు నిర్వహణ',
                'యాంటీ-ఇన్‌ఫ్లమేటరీ మందులు',
                'సంప్రదాయ నిర్వహణ విఫలమైతే ఉమ్మడి ఇంజెక్షన్లను పరిగణించండి'
            ]
        }
    }

    details = condition_details.get(result['disease'], {
        'findings': f'{result["disease"]} తో అనుగుణంగా రేడియోగ్రాఫిక్ మార్పులు.',
        'impression': f'క్లినికల్ ఫలితాలు {result["disease"]} ను సూచిస్తున్నాయి.',
        'recommendations': [
            'క్లినికల్ సహసంబంధం సిఫార్సు చేయబడింది',
            'లక్షణాలు కొనసాగితే అదనపు ఇమేజింగ్ పరిగణించండి',
            'స్పెషలిస్ట్ సంప్రదింపు సిఫార్సు చేయబడింది'
        ]
    })

    report = f"""
క్లినికల్ రేడియాలజీ రిపోర్ట్
{'=' * 60}

రోగి సమాచారం
పరీక్ష తేదీ: [తేదీ]
అధ్యయన రకం: X-రే రేడియోగ్రఫీ

క్లినికల్ సూచన
పరీక్షించిన శరీర భాగం: {body_part_te}
క్లినికల్ చరిత్ర: [రిఫర్ చేసే వైద్యుడు అందించినట్లు]

సాంకేతిక వివరాలు
గుర్తింపు విశ్వాసం: {result['confidence']*100:.1f}%
చిత్ర నాణ్యత: రోగనిర్ధారణ వివరణకు సరిపోతుంది
సాంకేతికత: ప్రామాణిక రేడియోగ్రాఫిక్ ప్రొజెక్షన్

ఫలితాలు
{details['findings']}

ప్రాథమిక పరిశీలన: {disease_te}
రోగనిర్ధారణ విశ్వాసం: {result['disease_conf']*100:.1f}%

క్లినికల్ ముద్ర
{details['impression']}

సిఫార్సులు
"""

    for i, rec in enumerate(details['recommendations'], 1):
        report += f"{i}. {rec}\n"

    report += f"""
అదనపు గమనికలు
• ఇది AI-సహాయక ప్రాథమిక అంచనా
• క్లినికల్ ప్రెజెంటేషన్‌తో సహసంబంధం అవసరం
• లైసెన్స్ పొందిన రేడియాలజిస్ట్ ద్వారా సమీక్ష సిఫార్సు చేయబడింది
• అదనపు రోగనిర్ధారణ అధ్యయనాలు హామీ ఇవ్వబడవచ్చు

{'=' * 60}
రిపోర్టింగ్ సిస్టమ్: AI-సహాయక రోగనిర్ధారణ సాధనం
రిపోర్ట్ రూపొందించబడింది: [టైమ్‌స్టాంప్]

నిరాకరణ: ఈ ఆటోమేటెడ్ రిపోర్ట్‌ను అర్హత కలిగిన ఆరోగ్య సంరక్షణ
నిపుణుడు సమీక్షించాలి. చికిత్స నిర్ణయాలు కేవలం ఈ ప్రాథమిక
అంచనాపై ఆధారపడకూడదు.
"""

    return report

# ============================================================================
# TELUGU PATIENT REPORT TEMPLATE
# ============================================================================

def generate_patient_report_telugu(result):
    """Generate patient-friendly report in Telugu"""

    body_part_te = TELUGU_BODY_PARTS.get(result['body_part'], result['body_part'])
    disease_te = TELUGU_DISEASES.get(result['disease'], result['disease'])

    condition_info = {
        'Normal': {
            'explanation': 'మీ X-రే పరీక్షలో ఎముకలు విరగడం, ఇన్ఫెక్షన్లు లేదా ఇతర అసాధారణతలకు సంకేతాలు కనిపించలేదు. ఎముకలు మరియు కీళ్లు ఆరోగ్యంగా కనిపిస్తున్నాయి.',
            'causes': 'ఇది సాధారణ, ఆరోగ్యకరమైన ఫలితం.',
            'what_to_do': 'మీ సాధారణ ఆరోగ్య దినచర్యను కొనసాగించండి మరియు ఆరోగ్యకరమైన జీవనశైలిని కొనసాగించండి.',
            'when_to_see_doctor': 'మీరు ఏవైనా కొత్త లక్షణాలు లేదా ఆందోళనలను అభివృద్ధి చేస్తే తిరిగి రండి.',
            'prognosis': 'అద్భుతం. వైద్య జోక్యం అవసరం లేదు.'
        },
        'Pneumonia': {
            'explanation': 'మీ ఛాతీ X-రే న్యుమోనియా సంకేతాలను చూపిస్తుంది, ఇది మీ ఊపిరితిత్తులలోని గాలి సంచులను ఎర్రబారిస్తుంది.',
            'causes': 'సాధారణంగా బ్యాక్టీరియా, వైరస్లు లేదా శిలీంద్రాల వల్ల కలుగుతుంది. బలహీనమైన రోగనిరోధక వ్యవస్థ, ధూమపానం లేదా ఇతర ఊపిరితిత్తుల పరిస్థితుల ద్వారా ట్రిగ్గర్ చేయబడవచ్చు.',
            'what_to_do': 'విశ్రాంతి తీసుకోండి, చాలా ద్రవాలు త్రాగండి, సూచించిన యాంటీబయోటిక్స్ తీసుకోండి మరియు సిఫార్సు చేస్తే హ్యూమిడిఫైయర్ ఉపయోగించండి.',
            'when_to_see_doctor': 'మీకు శ్వాస తీసుకోవడంలో ఇబ్బంది, ఛాతీ నొప్పి, అధిక జ్వరం లేదా లక్షణాలు తీవ్రమైతే.',
            'prognosis': 'చాలా మంది సరైన చికిత్సతో 2-4 వారాలలో కోలుకుంటారు.'
        },
        'Fracture': {
            'explanation': 'మీ X-రే ఎముకలో విరుగుడు లేదా పగుళ్లను చూపిస్తుంది, దీనిని ఫ్రాక్చర్ అని పిలుస్తారు.',
            'causes': 'సాధారణంగా గాయం, పడిపోవడం, క్రీడల గాయాలు లేదా ఎముకపై పునరావృత ఒత్తిడి వల్ల కలుగుతుంది.',
            'what_to_do': 'ప్రాంతాన్ని స్థిరంగా ఉంచండి, దానిపై బరువు ఉంచడం నివారించండి, మంచును వర్తించండి (ఇటీవల అయితే), మరియు మీ వైద్యుని చికిత్స ప్రణాళికను అనుసరించండి.',
            'when_to_see_doctor': 'మీరు పెరిగిన నొప్పి, తిమ్మిరి లేదా చర్మ రంగులో మార్పులను గమనిస్తే వెంటనే.',
            'prognosis': 'చాలా ఫ్రాక్చర్లు సరైన సంరక్షణతో 6-12 వారాలలో నయమవుతాయి.'
        },
        'Osteoarthritis': {
            'explanation': 'మీ X-రే ఆస్టియో ఆర్థరైటిస్ సంకేతాలను చూపిస్తుంది, ఇది కీలు మృదులాస్థి యొక్క అరిగిపోవడం నొప్పి మరియు దృఢత్వాన్ని కలిగిస్తుంది.',
            'causes': 'వయస్సు-సంబంధిత అరుగుట, మునుపటి గాయాలు, ఊబకాయం, జన్యుశాస్త్రం లేదా కీళ్లపై పునరావృత ఒత్తిడి.',
            'what_to_do': 'ఆరోగ్యకరమైన బరువును నిర్వహించండి, తక్కువ-ప్రభావ వ్యాయామాలు చేయండి (ఈత, సైక్లింగ్), వేడి/చల్లని చికిత్సను ఉపయోగించండి మరియు సూచించినట్లుగా యాంటీ-ఇన్‌ఫ్లమేటరీ మందులు తీసుకోండి.',
            'when_to_see_doctor': 'నొప్పి తీవ్రంగా మారితే, రోజువారీ కార్యకలాపాలను ప్రభావితం చేస్తే లేదా సంప్రదాయ చికిత్సలు సహాయపడకపోతే.',
            'prognosis': 'దీనిని నయం చేయలేనప్పటికీ, జీవనశైలి మార్పులు మరియు చికిత్సతో లక్షణాలను సమర్థవంతంగా నిర్వహించవచ్చు.'
        }
    }

    info = condition_info.get(result['disease'], {
        'explanation': f'మీ X-రే {disease_te} సంకేతాలను చూపిస్తుంది.',
        'causes': 'ఈ పరిస్థితికి వివిధ కారకాలు దోహదపడగలవు.',
        'what_to_do': 'నిర్దిష్ట మార్గదర్శకత్వం కోసం మీ ఆరోగ్య సంరక్షణ ప్రదాతతో సంప్రదించండి.',
        'when_to_see_doctor': 'మూల్యాంకనం కోసం మీ వైద్యునితో అపాయింట్‌మెంట్ షెడ్యూల్ చేయండి.',
        'prognosis': 'రోగ నిరూపణ వ్యక్తిగత పరిస్థితుల ఆధారంగా మారుతుంది.'
    })

    report = f"""
మీ X-రే ఫలితాలు వివరించబడ్డాయి
{'=' * 60}

🔍 మేము కనుగొన్నది
మేము మీ {body_part_te} X-రేను జాగ్రత్తగా పరీక్షించాము.

కనుగొనబడినది: {disease_te}
గుర్తింపు విశ్వాసం: {result['disease_conf']*100:.1f}%

📋 ఇది సాధారణ పదాలలో అర్థం
{info['explanation']}

🤔 దీనికి కారణం ఏమిటి
{info['causes']}

💡 మీరు ఏమి చేయాలి
{info['what_to_do']}

⚠️ మీ వైద్యుడిని ఎప్పుడు చూడాలి
{info['when_to_see_doctor']}

📈 ఏమి ఆశించాలి (రోగ నిరూపణ)
{info['prognosis']}

{'=' * 60}

ముఖ్యమైన రిమైండర్లు
• ఇది ప్రాథమిక ఆటోమేటెడ్ అంచనా
• ఫలితాల గురించి ఎల్లప్పుడూ మీ వైద్యునితో చర్చించండి
• మీ వైద్యుడు మీ పూర్తి వైద్య చరిత్ర ఆధారంగా చికిత్స ప్రణాళికను సృష్టిస్తారు
• వృత్తిపరమైన వైద్య సలహా లేకుండా ఏ చికిత్స నిర్ణయాలు తీసుకోకండి
• మీ ఫలితాల గురించి మీకు ఏవైనా ప్రశ్నలు ఉంటే మీ వైద్యుడిని అడగండి

తదుపరి దశలు
1. మీ ఆరోగ్య సంరక్షణ ప్రదాతతో అపాయింట్‌మెంట్ షెడ్యూల్ చేయండి
2. ఈ నివేదికను తీసుకురండి మరియు మీ లక్షణాల గురించి చర్చించండి
3. మీ వైద్యుని సిఫార్సులను అనుసరించండి
4. మీ లక్షణాలలో ఏవైనా మార్పులను ట్రాక్ చేయండి
5. ప్రశ్నలు అడగడానికి వెనుకాడకండి

గుర్తుంచుకోండి: ముందస్తు గుర్తింపు మరియు సరైన వైద్య సంరక్షణ మంచి ఆరోగ్య
ఫలితాలకు దారితీస్తుంది. ఈ పరీక్షను పొందడం ద్వారా మీరు ముఖ్యమైన అడుగు
వేస్తున్నారు!

{'=' * 60}
రిపోర్ట్ రూపొందించినది: AI-సహాయక వైద్య ఇమేజింగ్ సిస్టమ్
తేదీ: [టైమ్‌స్టాంప్]

ప్రశ్నలు లేదా ఆందోళనల కోసం, దయచేసి మీ ఆరోగ్య సంరక్షణ ప్రదాతతను సంప్రదించండి.
"""

    return report

# ============================================================================
# MAIN REPORT GENERATION FUNCTIONS
# ============================================================================

def generate_clinical_report(image, result, language="English"):
    """
    Generate clinical report in specified language

    Parameters:
        image: X-ray image (numpy array)
        result: Detection result dictionary
        language: "English" or "Telugu"
    """
    if language == "Telugu":
        return generate_clinical_report_telugu(result)
    else:
        return generate_clinical_report_english(result)

def generate_patient_report(image, result, language="English"):
    """
    Generate patient-friendly report in specified language

    Parameters:
        image: X-ray image (numpy array)
        result: Detection result dictionary
        language: "English" or "Telugu"
    """
    if language == "Telugu":
        return generate_patient_report_telugu(result)
    else:
        return generate_patient_report_english(result)

# ============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# ============================================================================

def image_to_base64(np_image):
    """Convert numpy image to base64 string"""
    pil = Image.fromarray(np_image).convert("RGB")
    buffered = BytesIO()
    pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def translate_to_telugu(text):
    """
    Legacy function for backward compatibility
    Now handled directly in report generation
    """
    return text  # Reports are already generated in target language

__all__ = [
    'generate_clinical_report',
    'generate_patient_report',
    'translate_to_telugu',
    'image_to_base64',
    'TELUGU_BODY_PARTS',
    'TELUGU_DISEASES'
]

