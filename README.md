# 📚 Arabic Book Recommendation System (TF-IDF + KNN)

This project is a **book recommendation system** specifically designed for **Arabic novels and literature**.  
It uses **TF-IDF vectorization** and **K-Nearest Neighbors (KNN)** with **cosine similarity** to suggest books that are similar in content, genre, and description.

## 🔍 What This Project Does

- Suggest similar Arabic books based on an input title.
- If the book is not found, you can input a new book manually (author, genre, page count, and description) to get similar books from the dataset.
- Simple interactive command-line interface (CLI).
- Built for Arabic language support and understanding.

## 📂 Dataset Information

> **Source**: The data was collected from [Aseer Al-Kotob], a well-known Arabic book website.

⚠️ Out of respect for copyright and fair use, **only a small sample** of the original dataset will be included in this repository.

You may replace `all_book.csv` with your own dataset in the following format:

| Title (عنوان الكتاب) | Author (المؤلف) | Category (التصنيف) | Page Count (عدد الصفحات) | Description (الوصف) |

## 💡 Key Features

- ✅ Designed for Arabic language.
- ✅ Uses TF-IDF to extract meaningful features from book descriptions.
- ✅ Uses KNN with cosine similarity to compute closest matches.
- ✅ Works for both existing and new (unlisted) books.
- ✅ Clean and modular code ready for deployment or further extension.

## 🛠️ How It Works

### 1. Data Preprocessing
- Converts page count to numeric format.
- Drops any rows with missing essential values.
- Concatenates `author + category + description` to create a `features` column.

### 2. Feature Extraction
- Applies `TfidfVectorizer` (with Arabic stopwords) to the `features` column.
- Applies `MinMaxScaler` to normalize page count.
- Combines text and numerical features using `scipy.sparse.hstack`.

### 3. Model Training
- Fits a KNN model using cosine similarity on the combined feature matrix.

### 4. Recommendation Logic
- For a known book: finds its vector and retrieves top 5 similar books.
- For a new book: vectorizes the input and uses the trained model to suggest similar books.

## 🖥️ How to Use

### 📦 Install Requirements
```bash
pip install -r requirements.txt
```

### ▶️ Run the Recommender
```bash
python recommender.py
```

### 🧑‍💻 Interface Flow

- Enter the name of an existing book to get recommendations.
- Type `جديد` to enter a new book manually.
- Type `توقف` to exit the program.

## 📁 Project Structure

book-recommender/
│
├── data/
│   └── all_book.csv        # CSV file with Arabic book data
│
├── recommender.py          # Main script for the recommendation engine
├── requirements.txt        # Python dependencies
├── README.md               # This documentation file

## 📝 Requirements

```txt
pandas
scikit-learn
scipy
```

## 🚀 Future Improvements

- Add GUI with Streamlit or Flask
- Integrate semantic embeddings (e.g., AraBERT)
- Automatically clean and normalize Arabic text
- Export recommendations to file or share via API

## ⚖️ License & Fair Use

This project is for **educational and research purposes only**.  
All data used belongs to its original copyright holders.

No commercial use is allowed unless permission is obtained from the data source.

## 📫 Contact

If you have suggestions or would like to contribute, feel free to fork the repository or open an issue.
