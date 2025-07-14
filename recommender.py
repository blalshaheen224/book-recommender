import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse

df = pd.read_csv("/content/all_book.csv")

# 1. تحويل عدد الصفحات إلى أرقام
df["عدد الصفحات"] = pd.to_numeric(df["عدد الصفحات"], errors='coerce')

# 2. حذف الصفوف اللي فيها قيم غير صالحة
df.dropna(subset=["عدد الصفحات"], inplace=True)

# 3. إعادة ضبط الفهرس
df.reset_index(drop=True, inplace=True)

df.dropna(subset=['عنوان الكتاب', 'المؤلف', 'التصنيف', 'عدد الصفحات', 'الوصف'], inplace=True)
df["features"] = df["المؤلف"] + " " + df["التصنيف"] + " " + df["الوصف"]
arabic_stopwords = [
    "في", "من", "إلى", "على", "أن", "عن", "ما", "مع", "هذا", "هذه",
    "كان", "لدي", "لقد", "إن", "ذلك", "ثم", "كل", "قد", "لا", "لم",
    "له", "فيها", "هو", "هي", "أو", "أكثر", "بين", "حتى", "إذا", "بعد"
    # ممكن تزود حسب الحاجة
]
vectorizer = TfidfVectorizer(stop_words=arabic_stopwords, max_features=1000)
text_features = vectorizer.fit_transform(df["features"])

scaler = MinMaxScaler()
pages_scaled = scaler.fit_transform(df[["عدد الصفحات"]])
X = scipy.sparse.hstack([text_features, pages_scaled])
X = X.tocsr()
model = NearestNeighbors(n_neighbors=6, metric='cosine')
model.fit(X)


def recommend_books(title):
    title = title.strip()
    if title not in df['عنوان الكتاب'].values:
        print("this book is not availability please enter this book maunally")
        return
    idx = df[df['عنوان الكتاب'] == title].index[0]

    distances, indices = model.kneighbors(X[idx])
    print(f"\n📚 الكتب المشابهة لـ '{title}':")
    for i in range(1, len(indices[0])):
        book_title = df.iloc[indices[0][i]][0]
        print(book_title)


# 9. تابع التعامل مع كتاب جديد
def recommend_new_book(author, category, pages, description):
    new_text = f"{author} {category} {description}"
    new_text_vec = vectorizer.transform([new_text])
    pages_scaled = scaler.transform([[pages]])
    new_vec = scipy.sparse.hstack([new_text_vec, pages_scaled])

    distances, indices = model.kneighbors(new_vec)
    print("\n📚 أقرب الكتب لهذا الوصف:")
    for i in range(5):
        print(f"- {df.iloc[indices[0][i]][0]}")


while True:
    print("\n📘 أدخل اسم كتاب (أو اكتب 'جديد' لإضافة كتاب غير موجود - أو 'توقف' للخروج):")
    user_input = input(">> ").strip()

    if user_input.lower() == "توقف":
        print("👋 تم إيقاف البرنامج.")
        break

    elif user_input.lower() == "جديد":
        print("✍️ أدخل معلومات الكتاب الجديد:")
        author = input("اسم المؤلف: ").strip()
        category = input("التصنيف: ").strip()

        # التحقق من صحة عدد الصفحات
        while True:
            try:
                pages = int(input("عدد الصفحات: ").strip())
                break
            except ValueError:
                print("⚠️ من فضلك أدخل رقم صحيح لعدد الصفحات.")

        description = input("الوصف: ").strip()
        recommend_new_book(author, category, pages, description)

    else:
        recommend_books(user_input)

