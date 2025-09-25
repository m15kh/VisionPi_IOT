"""
This module contains prompt templates for the LLM model.
Templates are organized by language and functionality.
"""

# Command detection prompts
LAMP_COMMAND_PROMPTS = {
    # English prompt for lamp command detection
    "en": """Analyze whether the following text contains a command to turn a lamp/light/LED on or off:
"{text}"

Consider different ways users might refer to lights (lamp, light, LED, bulb, lighting, etc.) 
and different ways they might express the command (turn on, switch on, activate, power on, etc.).

If it's a request to turn on a light/lamp/LED, respond with: "on"
If it's a request to turn off a light/lamp/LED, respond with: "off"
If there's no clear command about a light/lamp/LED, respond with: "unknown"

Decision:""",

    # Persian prompt for lamp command detection
    "fa": """تحلیل کنید که آیا متن زیر شامل دستوری برای روشن یا خاموش کردن چراغ، لامپ یا LED است:
"{text}"

راه‌های مختلفی که ممکن است کاربران به چراغ اشاره کنند (چراغ، لامپ، LED، روشنایی، نور و غیره) و 
روش‌های مختلفی که ممکن است دستور را بیان کنند (روشن کردن، فعال کردن، زدن و غیره) را در نظر بگیرید.

اگر درخواست روشن کردن چراغ/لامپ/LED است، پاسخ دهید: "on"
اگر درخواست خاموش کردن چراغ/لامپ/LED است، پاسخ دهید: "off"
اگر هیچ دستور مشخصی برای چراغ/لامپ/LED نیست، پاسخ دهید: "unknown"

تصمیم:"""
}
