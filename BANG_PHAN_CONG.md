# BẢNG PHÂN CÔNG CÔNG VIỆC

**Đề tài:** Phân tích và dự đoán giá bất động sản tại Hà Nội kết hợp Machine Learning
**Nhóm:** Nhóm 01 - Lập trình Python
**GVHD:** Lương Ngọc Sơn
**Tổng số nhiệm vụ:** 33

## 1. Thành viên và tỷ lệ đóng góp

| STT | Họ và tên | MSSV | Mảng phụ trách | Mã task | Số task | Tỷ lệ |
|---:|---|---|---|---|---:|---:|
| 1 | Tô Thành Công | B25DCTC024 | Machine Learning | ML01–ML05 | 5 | 15,2% |
| 2 | Bùi Quốc Anh | B25DCTC004 | Báo cáo & thuyết trình | R01–R07 | 7 | 21,2% |
| 3 | Đặng Đức Anh | B25DCTC013 | Phân tích thị trường | M01–M08 | 8 | 24,2% |
| 4 | Trần Hoàng Dũng | B25DCTC030 | EDA & biểu đồ | E01–E07 | 7 | 21,2% |
| 5 | Dương Minh Tuấn | B25DCTC132 | Xử lý dữ liệu | D01–D06 | 6 | 18,2% |

## 2. Chi tiết nhiệm vụ

### Dương Minh Tuấn — B25DCTC132 — Xử lý dữ liệu (D01–D06)

| Mã | Nhiệm vụ | Sản phẩm bàn giao |
|---|---|---|
| D01 | Kiểm tra cột, kiểu dữ liệu và ý nghĩa từng cột | Bảng từ điển dữ liệu |
| D02 | Thống kê số dòng ở từng bước lọc | Bảng pipeline: toàn quốc → Hà Nội → hợp lệ |
| D03 | Kiểm tra missing, trùng lặp, giá trị bất thường | Báo cáo chất lượng dữ liệu |
| D04 | Thống kê số tin, tỷ lệ, giá trung vị, diện tích trung vị theo loại hình | Bảng thống kê mô tả theo phân khúc |
| D05 | Kiểm tra tin có giá hoặc giá/m² quá cao | Danh mục kiểm định ngoại lai |
| D06 | Bàn giao từ điển dữ liệu và file thống kê dữ liệu sạch | `data_dictionary.xlsx`, CSV thống kê |

### Trần Hoàng Dũng — B25DCTC030 — EDA & biểu đồ (E01–E07)

| Mã | Nhiệm vụ | Sản phẩm bàn giao |
|---|---|---|
| E01 | Biểu đồ phân phối giá, diện tích, giá/m² | Histogram / KDE 3 biến |
| E02 | Biểu đồ phân phối ở thang log | Log-scale distribution plots |
| E03 | So sánh giá giữa Nhà, Đất, Căn hộ, Shophouse | Biểu đồ so sánh phân khúc |
| E04 | Boxplot theo loại hình và theo quận/huyện | Bộ boxplot phân cấp |
| E05 | Scatter diện tích và giá kèm đường xu hướng | Biểu đồ tán xạ |
| E06 | Heatmap ma trận tương quan | Correlation heatmap |
| E07 | Nhận xét chi tiết cho từng biểu đồ | Tài liệu chú giải và phân tích |

### Đặng Đức Anh — B25DCTC013 — Phân tích thị trường (M01–M08)

| Mã | Nhiệm vụ | Sản phẩm bàn giao |
|---|---|---|
| M01 | So sánh loại hình theo số tin, giá trung vị, giá/m², diện tích | Bảng tổng hợp đa chiều |
| M02 | So sánh giá giữa các quận/huyện, xếp hạng giá | Báo cáo phân tích không gian |
| M03 | Hệ số tương quan Pearson và Spearman giữa diện tích và giá | Bảng tương quan kèm p-value |
| M04 | Loại hình nào có giá/m² cao nhất | Báo cáo insight chuyên đề |
| M05 | Khu vực nào có giá trung vị cao nhất | Báo cáo insight địa bàn |
| M06 | Giá thay đổi theo diện tích như thế nào | Đánh giá độ nhạy giá theo quy mô |
| M07 | Loại hình nào thiếu dữ liệu nhiều nhất | Bảng tỷ lệ khuyết thiếu theo phân khúc |
| M08 | Nhận xét dựa trên số liệu thực tế khách quan | Bản nhận định thị trường |

### Bùi Quốc Anh — B25DCTC004 — Báo cáo & thuyết trình (R01–R07)

| Mã | Nhiệm vụ | Sản phẩm bàn giao |
|---|---|---|
| R01 | Ghép nội dung thành báo cáo theo bố cục chuẩn | Báo cáo tổng hợp |
| R02 | Chuẩn hóa thuật ngữ toàn tài liệu và slide | Bản rà soát thuật ngữ |
| R03 | Thiết kế slide / bản tóm tắt dự án | Bộ thuyết trình |
| R04 | Đưa kết quả ML vào bảng so sánh mô hình | Bảng so sánh hiệu năng |
| R05 | Kiểm tra chéo số liệu báo cáo với file output gốc | Biên bản đối soát số liệu |
| R06 | Danh mục tài liệu tham khảo theo APA/IEEE | Danh mục tham khảo |
| R07 | Bộ câu hỏi phản biện và dự thảo trả lời | Tài liệu Q&A |

### Tô Thành Công — B25DCTC024 — Machine Learning (ML01–ML05)

| Mã | Nhiệm vụ | Sản phẩm bàn giao |
|---|---|---|
| ML01 | Tiền xử lý đầu vào phục vụ huấn luyện | Feature pipeline |
| ML02 | Chia train/test theo nhóm tin đăng (Group-based split) | Pipeline phân chia tập |
| ML03 | Huấn luyện và đánh giá các mô hình dự báo giá | Mô hình ML kèm kết quả đánh giá |
| ML04 | Kiểm tra rò rỉ dữ liệu toàn bộ khâu mô hình | Báo cáo kiểm soát leakage |
| ML05 | Gửi metrics và kết luận ML | `metrics.csv`, `metrics_by_property_type.csv` |

## 3. Ghi chú

- Tỷ lệ tính theo số nhiệm vụ đảm nhận trên tổng 33 nhiệm vụ.
- Toàn bộ 33 nhiệm vụ đã hoàn thành; sản phẩm tương ứng nằm trong gói nộp.
- Mọi thành viên tham gia kiểm tra chéo kết quả trước khi nộp.

| Họ và tên | Chữ ký |
|---|---|
| Tô Thành Công | |
| Bùi Quốc Anh | |
| Đặng Đức Anh | |
| Trần Hoàng Dũng | |
| Dương Minh Tuấn | |
