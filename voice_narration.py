

import os
import re
from gtts import gTTS
import tempfile
from pathlib import Path


class VoiceNarrator:
    """
    Generates voice narration for medical reports in multiple languages.
    Cleans text to remove symbols, formulas, and special characters for natural speech.
    """

    # Language codes supported by gTTS
    SUPPORTED_LANGUAGES = {
        "English": "en",
        "Telugu": "te",
        "Hindi": "hi",
        "Tamil": "ta",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Arabic": "ar",
        "Chinese": "zh-cn",
        "Japanese": "ja",
        "Korean": "ko",
        "Portuguese": "pt",
        "Russian": "ru",
        "Italian": "it"
    }

    def __init__(self):
        """Initialize the voice narrator"""
        self.output_dir = "/tmp/xray_audio"
        os.makedirs(self.output_dir, exist_ok=True)
        print("🔊 Voice Narrator initialized")

    def clean_text_for_narration(self, text):
        """
        Clean text for natural speech narration.
        Removes symbols, formulas, special characters, and formatting.

        Parameters
        ----------
        text : str
            Raw text from the report

        Returns
        -------
        str : Cleaned text suitable for narration
        """
        # Remove common symbols and formatting markers
        text = re.sub(r'[🔬📊🔄🗺️📐✅❌⚠️🎯📈📉💡🏥🔥📦📝📥📄🎤🔊🌐]', '', text)  # Remove emojis
        text = re.sub(r'[●•▪▫■□◆◇○◉]', '', text)  # Remove bullet points
        text = re.sub(r'[-─━═]{3,}', '', text)  # Remove horizontal lines (---, ===, etc.)
        text = re.sub(r'[*#_]{2,}', '', text)  # Remove markdown bold/italic markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove markdown bold but keep text
        text = re.sub(r'__([^_]+)__', r'\1', text)  # Remove underline markdown
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove code backticks

        # Remove section separators and decorative lines
        text = re.sub(r'^={3,}.*$', '', text, flags=re.MULTILINE)  # Remove lines like ====
        text = re.sub(r'^\*{3,}.*$', '', text, flags=re.MULTILINE)  # Remove lines like ****

        # Convert percentages to spoken form
        text = re.sub(r'(\d+\.?\d*)%', r'\1 percent', text)

        # Convert common medical abbreviations to full words
        abbreviations = {
            'mm': 'millimeters',
            'cm': 'centimeters',
            'px': 'pixels',
            'vs': 'versus',
            'w/': 'with',
            'w/o': 'without',
            'approx.': 'approximately',
            'min': 'minimum',
            'max': 'maximum',
            'avg': 'average',
        }
        for abbr, full in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text, flags=re.IGNORECASE)

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?\-()]', ' ', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = ' '.join(lines)

        # Remove very short fragments (less than 3 characters)
        words = text.split()
        words = [w for w in words if len(w) >= 3 or w in ['.', ',', '!', '?']]
        text = ' '.join(words)

        return text.strip()

    def extract_sentences_and_values(self, text):
        """
        Extract only meaningful sentences and numerical values.
        Removes headers, labels, and formatting.

        Parameters
        ----------
        text : str
            Report text

        Returns
        -------
        str : Clean sentences with values
        """
        lines = text.split('\n')
        narrative_lines = []

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip section headers (all caps, short lines)
            if line.isupper() and len(line) < 50:
                continue

            # Skip lines that are just labels (ending with :)
            if line.endswith(':') and len(line.split()) <= 5:
                continue

            # Skip lines with only symbols
            if re.match(r'^[^\w\s]+$', line):
                continue

            # Convert bullet points to sentences
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                line = line[1:].strip()
                if line and not line.endswith('.'):
                    line += '.'

            # Keep lines that have actual content (sentences with verbs or meaningful info)
            if len(line.split()) >= 3:  # At least 3 words
                narrative_lines.append(line)

        return ' '.join(narrative_lines)

    def prepare_text_for_speech(self, text):
        """
        Complete text preparation pipeline for natural speech.

        Parameters
        ----------
        text : str
            Raw report text

        Returns
        -------
        str : Speech-ready text
        """
        # Step 1: Extract meaningful content
        text = self.extract_sentences_and_values(text)

        # Step 2: Clean symbols and formatting
        text = self.clean_text_for_narration(text)

        # Step 3: Add natural pauses
        # Replace multiple periods with a single period for better speech flow
        text = re.sub(r'\.{2,}', '.', text)

        # Ensure proper sentence endings
        text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)

        return text

    def generate_narration(self, text, language="English", slow=False):
        """
        Generate audio narration from text in specified language.
        Automatically cleans text for natural speech.

        Parameters
        ----------
        text : str
            Text to narrate
        language : str
            Language name (e.g., "English", "Telugu", "Hindi")
        slow : bool
            Whether to speak slowly (default: False)

        Returns
        -------
        str : Path to generated audio file, or None if failed
        """
        try:
            # Get language code
            lang_code = self.SUPPORTED_LANGUAGES.get(language, "en")

            print(f"🎤 Generating {language} narration...")

            # Clean text for speech (only for full reports, not summaries)
            if len(text) > 200:  # Assume longer text is a full report
                print("🧹 Cleaning text for natural speech...")
                original_length = len(text)
                text = self.prepare_text_for_speech(text)
                print(f"   Reduced from {original_length} to {len(text)} characters")

            # Create TTS object
            tts = gTTS(text=text, lang=lang_code, slow=slow)

            # Generate unique filename
            audio_filename = f"xray_report_{language.lower()}_{hash(text) % 100000}.mp3"
            audio_path = os.path.join(self.output_dir, audio_filename)

            # Save audio file
            tts.save(audio_path)

            print(f"✅ Audio saved: {audio_path}")
            return audio_path

        except Exception as e:
            print(f"❌ Error generating narration: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_summary_narration(self, body_part, disease, confidence, language="English"):
        """
        Generate a brief summary narration of the key findings.
        """
        # Create summary text based on language
        summaries = {
            "English": f"X-ray analysis complete. Body part detected: {body_part}. Primary finding: {disease}. Confidence level: {confidence*100:.1f} percent. Please review the detailed report for complete information.",

            "Telugu": f"ఎక్స్-రే విశ్లేషణ పూర్తయింది. గుర్తించిన శరీర భాగం: {body_part}. ప్రధాన కనుగొనబడినది: {disease}. విశ్వాస స్థాయి: {confidence*100:.1f} శాతం. పూర్తి సమాచారం కోసం దయచేసి వివరణాత్మక నివేదికను సమీక్షించండి.",

            "Hindi": f"एक्स-रे विश्लेषण पूर्ण हुआ। पहचाना गया शरीर का अंग: {body_part}। प्राथमिक निष्कर्ष: {disease}। विश्वास स्तर: {confidence*100:.1f} प्रतिशत। पूर्ण जानकारी के लिए कृपया विस्तृत रिपोर्ट की समीक्षा करें।",

            "Tamil": f"எக்ஸ்-ரே பகுப்பாய்வு முடிந்தது. கண்டறியப்பட்ட உடல் பகுதி: {body_part}. முதன்மை கண்டுபிடிப்பு: {disease}. நம்பிக்கை நிலை: {confidence*100:.1f} சதவீதம். முழு தகவலுக்கு விரிவான அறிக்கையை மதிப்பாய்வு செய்யவும்.",

            "Spanish": f"Análisis de rayos X completo. Parte del cuerpo detectada: {body_part}. Hallazgo principal: {disease}. Nivel de confianza: {confidence*100:.1f} por ciento. Revise el informe detallado para obtener información completa.",

            "French": f"Analyse radiographique terminée. Partie du corps détectée: {body_part}. Constatation principale: {disease}. Niveau de confiance: {confidence*100:.1f} pour cent. Veuillez consulter le rapport détaillé pour des informations complètes.",

            "German": f"Röntgenanalyse abgeschlossen. Erkannter Körperteil: {body_part}. Hauptbefund: {disease}. Vertrauensniveau: {confidence*100:.1f} Prozent. Bitte überprüfen Sie den detaillierten Bericht für vollständige Informationen.",

            "Arabic": f"اكتمل تحليل الأشعة السينية. جزء الجسم المكتشف: {body_part}. النتيجة الأولية: {disease}. مستوى الثقة: {confidence*100:.1f} بالمائة. يرجى مراجعة التقرير المفصل للحصول على معلومات كاملة.",

            "Chinese": f"X光分析完成。检测到的身体部位：{body_part}。主要发现：{disease}。置信度：{confidence*100:.1f}%。请查看详细报告以获取完整信息。",

            "Japanese": f"X線分析が完了しました。検出された身体部位：{body_part}。主な所見：{disease}。信頼度：{confidence*100:.1f}パーセント。完全な情報については、詳細レポートをご確認ください。",

            "Korean": f"엑스레이 분석 완료. 감지된 신체 부위: {body_part}. 주요 발견: {disease}. 신뢰 수준: {confidence*100:.1f} 퍼센트. 전체 정보는 상세 보고서를 검토하십시오.",

            "Portuguese": f"Análise de raio-X concluída. Parte do corpo detectada: {body_part}. Descoberta principal: {disease}. Nível de confiança: {confidence*100:.1f} por cento. Reveja o relatório detalhado para informações completas.",

            "Russian": f"Рентгеновский анализ завершен. Обнаруженная часть тела: {body_part}. Основное заключение: {disease}. Уровень достоверности: {confidence*100:.1f} процентов. Пожалуйста, ознакомьтесь с подробным отчетом для получения полной информации.",

            "Italian": f"Analisi radiografica completata. Parte del corpo rilevata: {body_part}. Riscontro principale: {disease}. Livello di confidenza: {confidence*100:.1f} per cento. Si prega di rivedere il rapporto dettagliato per informazioni complete."
        }

        summary = summaries.get(language, summaries["English"])
        return self.generate_narration(summary, language)

    def get_supported_languages(self):
        """Get list of supported languages."""
        return list(self.SUPPORTED_LANGUAGES.keys())


def create_narration_for_report(report_text, language="English", include_summary=True,
                                 body_part=None, disease=None, confidence=None):
    """
    Convenience function to create narration from a report.
    Automatically cleans text for natural speech output.

    Parameters
    ----------
    report_text : str
        Full report text to narrate
    language : str
        Language for narration
    include_summary : bool
        Whether to create a summary narration in addition to full report
    body_part : str, optional
        Body part for summary
    disease : str, optional
        Disease for summary
    confidence : float, optional
        Confidence for summary

    Returns
    -------
    dict : Dictionary with 'full_narration' and optionally 'summary_narration' paths
    """
    narrator = VoiceNarrator()
    result = {}

    # Generate full report narration (will be automatically cleaned)
    full_audio = narrator.generate_narration(report_text, language)
    result['full_narration'] = full_audio

    # Generate summary narration if requested
    if include_summary and body_part and disease and confidence is not None:
        summary_audio = narrator.generate_summary_narration(
            body_part, disease, confidence, language
        )
        result['summary_narration'] = summary_audio

    return result


# Utility function for testing text cleaning
def preview_cleaned_text(report_text):
    """
    Preview how text will be cleaned for narration.
    Useful for debugging and testing.

    Parameters
    ----------
    report_text : str
        Original report text

    Returns
    -------
    str : Cleaned text that will be narrated
    """
    narrator = VoiceNarrator()
    cleaned = narrator.prepare_text_for_speech(report_text)

    print("=" * 80)
    print("ORIGINAL TEXT:")
    print("=" * 80)
    print(report_text[:500])
    print("\n" + "=" * 80)
    print("CLEANED TEXT FOR NARRATION:")
    print("=" * 80)
    print(cleaned[:500])
    print("=" * 80)
    print(f"Original length: {len(report_text)} characters")
    print(f"Cleaned length: {len(cleaned)} characters")
    print(f"Reduction: {((len(report_text) - len(cleaned)) / len(report_text) * 100):.1f}%")

    return cleaned

