import pandas as pd
import os

def preprocess_groceries(input_file="examples/Groceries_dataset.csv", output_file="examples/data_groceries.csv"):
    """
    Chuẩn hóa dữ liệu từ file Groceries.csv của Kaggle sang định dạng
    có thể dùng trực tiếp cho thuật toán FP-Growth.
    """

    # 1️⃣ Kiểm tra file đầu vào có tồn tại không
    if not os.path.exists(input_file):
        print(f"❌ Không tìm thấy file {input_file}. Hãy đảm bảo file nằm cùng thư mục với script này.")
        return

    print(f"📂 Đang đọc file: {input_file} ...")
    df = pd.read_csv(input_file)

    # 2️⃣ Kiểm tra các cột cần thiết
    required_columns = {'Member_number', 'Date', 'itemDescription'}
    if not required_columns.issubset(df.columns):
        print(f"❌ File CSV phải có 3 cột: {', '.join(required_columns)}")
        print(f"➡ Các cột hiện tại: {list(df.columns)}")
        return

    # 3️⃣ Gom nhóm các sản phẩm theo Member_number + Date
    print("🔄 Đang gom nhóm các giao dịch...")
    transactions = (
        df.groupby(['Member_number', 'Date'])['itemDescription']
        .apply(list)
        .tolist()
    )

    # 4️⃣ Ghi dữ liệu ra file CSV cho FP-Growth
    print(f"💾 Đang ghi dữ liệu ra file: {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for transaction in transactions:
            f.write(','.join(map(str, transaction)) + '\n')

    print(f"✅ Đã tạo file {output_file} — sẵn sàng cho FP-Growth!")
    print(f"📊 Tổng số giao dịch: {len(transactions)}")

if __name__ == "__main__":
    preprocess_groceries()
