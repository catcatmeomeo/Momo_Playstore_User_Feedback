# Phân tích ý kiến người dùng ví MoMo trên Play Store

## 1. Giới thiệu dự án
Dự án này tập trung vào việc thu thập (crawl data) và phân tích chuyên sâu các đánh giá, phản hồi của người dùng về ứng dụng ví điện tử MoMo trên nền tảng Google Play Store. Mục tiêu chính là trích xuất các insight về trải nghiệm người dùng, các lỗi hệ thống thường gặp và đánh giá mức độ hài lòng chung.

## 2. Cấu trúc thư mục dự án
- `Analyze_Insight_Momo_wallet.ipynb`: File Jupyter Notebook chứa toàn bộ mã nguồn xử lý, làm sạch dữ liệu, phân tích và trực quan hóa (visualization).
- `.gitignore`: Cấu hình ẩn các tệp tin hệ thống và các file dữ liệu thô `.xlsx` có dung lượng lớn nhằm tối ưu hóa kho lưu trữ.

## 3. Công nghệ và Thư viện sử dụng
- **Ngôn ngữ:** Python
- **Môi trường:** Google Colab / VS Code (Jupyter Notebook)
- **Thư viện chính:** Pandas, NumPy, Matplotlib, Seaborn...

## 4. Hướng dẫn chạy dự án trên máy cục bộ (Local)
1. Clone dự án về máy:
   ```bash
   git clone https://github.com
   ```
2. Đảm bảo bạn đã cài đặt Python và các thư viện cần thiết (`pip install pandas numpy matplotlib seaborn notebook`).
3. Chuẩn bị các file dữ liệu đầu vào (`MOMO.xlsx` và `MOMO (crawl data second).xlsx`) đặt cùng cấp thư mục với file notebook.
4. Mở VS Code hoặc Jupyter Notebook để chạy file `Analyze_Insight_Momo_wallet.ipynb`.
