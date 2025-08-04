from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='auto', target='ru')
translation = translator.translate("Hello, world!")
print(translation)