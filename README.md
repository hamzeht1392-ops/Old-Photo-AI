# 🖼️ Old Photo AI

### AI-Powered Photo Colorization & Intelligent Object Removal

**Old Photo AI** is an AI-powered image processing project designed to restore and enhance old photos.

The project can:

* 🎨 Colorize old black-and-white photos
* 🧹 Detect and remove unwanted objects
* ✨ Run the complete processing pipeline
* 🗣️ Accept Persian object-removal commands such as `ماشین را حذف کن`
* 💾 Save generated results as PNG files

The project uses a combination of modern AI models:

* **DDColor** — image colorization
* **Grounding DINO** — text-guided object detection
* **SAM 2** — object segmentation
* **LaMa** — image inpainting / object removal
* **Gradio** — user interface

The main notebook is designed to run in **Google Colab with GPU acceleration**.

---

# 🇮🇷 توضیحات فارسی

## 🖼️ Old Photo AI چیست؟

**Old Photo AI** یک پروژه هوش مصنوعی برای بازسازی و پردازش عکس‌های قدیمی است.

این پروژه می‌تواند:

* 🎨 عکس‌های سیاه‌وسفید را رنگی کند
* 🧹 اشیای ناخواسته را از داخل تصویر حذف کند
* ✨ فرایند کامل حذف شیء و رنگی‌سازی را انجام دهد
* 🗣️ دستورات فارسی برای حذف اشیا دریافت کند
* 💾 نتیجه پردازش را به صورت فایل PNG ذخیره کند

### موتورهای هوش مصنوعی پروژه

**DDColor**
برای رنگی‌سازی تصاویر قدیمی استفاده می‌شود.

**Grounding DINO**
برای پیدا کردن شیء موردنظر بر اساس متن استفاده می‌شود.

**SAM 2**
برای ساخت ماسک دقیق شیء شناسایی‌شده استفاده می‌شود.

**LaMa**
برای بازسازی ناحیه‌ای که شیء از آن حذف شده استفاده می‌شود.

**Gradio**
رابط کاربری پروژه را فراهم می‌کند.

---

# 🚀 اجرای پروژه در Google Colab

## روش پیشنهادی

فایل زیر را در Google Colab باز کنید:

`Old_Photo_AI_Colab.ipynb`

سپس:

1. وارد Google Colab شوید.
2. نوت‌بوک را باز کنید.
3. از مسیر:

`Runtime → Change runtime type`

یک محیط دارای **GPU** انتخاب کنید.

4. سلول‌های نوت‌بوک را از بالا به پایین اجرا کنید.
5. صبر کنید کتابخانه‌ها و مدل‌های موردنیاز دانلود و آماده شوند.
6. تصویر موردنظر خود را وارد کنید.
7. یکی از عملیات پردازش را اجرا کنید.

> اجرای مدل‌های این پروژه بدون GPU ممکن است بسیار کندتر باشد.

---

# 🎨 امکانات

## 1. Colorize Image

با استفاده از DDColor، تصاویر سیاه‌وسفید و قدیمی به تصاویر رنگی تبدیل می‌شوند.

## 2. Remove Object

کاربر می‌تواند مشخص کند چه شیئی باید از تصویر حذف شود.

نمونه دستورات فارسی:

```text
ماشین را حذف کن
سگ را حذف کن
درخت را حذف کن
آدم را حذف کن
صندلی را حذف کن
```

سیستم ابتدا شیء را با **Grounding DINO** پیدا می‌کند، سپس با **SAM 2** ماسک می‌سازد و در نهایت با **LaMa** ناحیه حذف‌شده را بازسازی می‌کند.

## 3. Full Processing

در حالت پردازش کامل، ابتدا شیء موردنظر حذف می‌شود و سپس تصویر رنگی می‌شود.

---

# 🧠 Project Pipeline

```text
Input Image
     │
     ▼
Grounding DINO
     │
     ▼
Object Detection
     │
     ▼
SAM 2
     │
     ▼
Object Mask
     │
     ▼
LaMa Inpainting
     │
     ▼
DDColor
     │
     ▼
Final Image
```

---

# 📦 Requirements

نسخه‌های اصلی استفاده‌شده در محیط پروژه:

```text
transformers==5.17.0
huggingface-hub==1.32.0
accelerate==1.15.0
gradio==6.28.0
pillow
tqdm
opencv-python-headless
numpy
simple-lama-inpainting==0.1.2
```

نوت‌بوک Colab نیز مراحل نصب این وابستگی‌ها را درون خودش دارد.

---

# 🤖 AI Models

The project downloads the required model weights automatically when they are first used.

Models used by the project include:

```text
piddnad/ddcolor_paper_tiny
IDEA-Research/grounding-dino-tiny
facebook/sam2.1-hiera-tiny
```

The DDColor source repository is also cloned automatically when necessary.

---

# 💾 Output

نتایج پردازش‌شده در محیط Colab در پوشه زیر ذخیره می‌شوند:

```text
/content/old_photo_ai_outputs
```

خروجی‌ها به صورت فایل PNG ذخیره می‌شوند.

---

# 🖥️ Local Application

فایل `app.py` یک رابط Gradio مستقل برای اجرای پروژه خارج از نوت‌بوک فراهم می‌کند.

برای اجرای آن در یک محیط Python سازگار:

```bash
pip install -r requirements.txt
python app.py
```

> مسیر اصلی و پیشنهادی پروژه برای کاربران عمومی، اجرای نوت‌بوک در Google Colab است.

---

# 📁 Project Structure

```text
Old-Photo-AI/
│
├── Old_Photo_AI_Colab.ipynb   # Google Colab notebook
├── app.py                     # Gradio application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # Project license
├── .gitignore                  # Ignored files
│
└── outputs/
    └── .gitkeep
```

---

# ⚠️ Notes

* اجرای پروژه به منابع محاسباتی قابل‌توجهی نیاز دارد.
* استفاده از GPU در Google Colab توصیه می‌شود.
* مدل‌ها در اولین اجرا دانلود می‌شوند و ممکن است زمان و فضای قابل‌توجهی مصرف کنند.
* کیفیت تشخیص و حذف شیء به کیفیت تصویر، نوع شیء و مدل‌های مورد استفاده بستگی دارد.
* نتیجه نهایی برای تمام تصاویر تضمین‌شده نیست.

---

# 🙌 Credits

### Created by **Armin Hamzeh**

AI-powered image processing project using:

**DDColor • Grounding DINO • SAM 2 • LaMa • Gradio**

---

# 📄 License

This project is distributed under the license included in the `LICENSE` file.

---

# ⭐ Support

اگر پروژه برایتان مفید بود، می‌توانید به آن ⭐ Star بدهید و آن را با دیگران به اشتراک بگذارید.

**Made with AI & Python**
