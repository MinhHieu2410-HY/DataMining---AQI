# AQI-Prediction

## Giới thiệu đề tài

* **Bài toán:** Dự đoán mức độ chất lượng không khí (AQI Category) tại Hà Nội dựa trên các thông số môi trường.
* **Mục tiêu:** Xây dựng mô hình Machine Learning để phân loại mức độ ô nhiễm và triển khai giao diện dự đoán bằng Streamlit.
* **Ý nghĩa:** Hỗ trợ cảnh báo sớm chất lượng không khí, góp phần bảo vệ sức khỏe cộng đồng.

---

## Dataset

* **Nguồn dữ liệu:** Hanoi AQI Weather Dataset.
* **Dữ liệu gồm:** AQI và các thông số thời tiết như CO, NO₂, SO₂, O₃, nhiệt độ, độ ẩm, áp suất,...
* **Nhãn dự đoán:**

  * Good
  * Moderate
  * Unhealthy for Sensitive Groups
  * Unhealthy
  * Very Unhealthy
  * Hazardous

---

## Pipeline

1. Đọc dữ liệu.
2. Chuyển AQI thành các mức chất lượng không khí.
3. Loại bỏ các cột không cần thiết.
4. Xử lý dữ liệu thiếu.
5. Label Encoding.
6. Chia Train/Test (80/20).
7. Huấn luyện **Random Forest Classifier**.
8. Dự đoán mức AQI từ dữ liệu người dùng nhập.

---

## Mô hình sử dụng

### Random Forest Classifier

**Lý do lựa chọn**

* Hoạt động tốt trên dữ liệu phi tuyến.
* Hạn chế overfitting.
* Độ chính xác cao với dữ liệu môi trường.
* Không cần chuẩn hóa dữ liệu trước khi huấn luyện.

---

## Hướng dẫn chạy

Cài thư viện:

```bash
pip install -r requirements.txt
```

Chạy ứng dụng:

```bash
streamlit run app.py
```

---

## Cấu trúc thư mục

```text
AQI-Prediction/
│
├── app.py
├── hanoi-aqi-weather-data.csv
├── requirements.txt
├── README.md
└── .gitignore
```
