import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cấu hình giao diện (Giao diện Darkmode hiện đại)
st.set_page_config(page_title="Hệ Thống Dự Đoán Khách Hàng", page_icon="🎯", layout="wide")

# Nhúng CSS Custom để làm giao diện xịn sò hơn (Gradient, bo góc, hiệu ứng hover)
st.markdown("""
<style>
    h1 {
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
        text-align: center;
        font-weight: 800;
        padding-bottom: 10px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease 0s;
        font-size: 18px;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(255, 65, 108, 0.4);
    }
    /* Đóng khung các khu vực nhập liệu */
    div[data-testid="column"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)

# TỪ ĐIỂN DỊCH (Giao diện Tiếng Việt -> Backend Tiếng Anh để AI hiểu)
map_yes_no = {"Có": "Yes", "Không": "No"}
map_gender = {"Nam": "Male", "Nữ": "Female"}
map_senior = {"Có": 1, "Không": 0}
map_multiple_lines = {"Không dùng ĐT bàn": "No phone service", "Không": "No", "Có": "Yes"}
map_internet = {"Cáp quang (Fiber optic)": "Fiber optic", "Cáp đồng (DSL)": "DSL", "Không dùng mạng": "No"}
map_service = {"Không dùng mạng": "No internet service", "Không": "No", "Có": "Yes"}
map_contract = {"Trả từng tháng": "Month-to-month", "Ký 1 năm": "One year", "Ký 2 năm": "Two year"}
map_payment = {
    "Sổ séc điện tử (Electronic check)": "Electronic check",
    "Sổ séc qua bưu điện (Mailed check)": "Mailed check",
    "Chuyển khoản tự động": "Bank transfer (automatic)",
    "Thẻ tín dụng tự động": "Credit card (automatic)"
}

@st.cache_resource
def load_components():
    preprocessor = joblib.load('preprocessor.pkl')
    model = joblib.load('catboost_model.pkl')
    return preprocessor, model

try:
    preprocessor, model = load_components()
except Exception as e:
    st.error("Lỗi: Không tìm thấy mô hình. Hãy chạy file train_model.py trước.")
    st.stop()

st.title("🎯 DỰ ĐOÁN XÁC SUẤT KHÁCH HÀNG RỜI BỎ")
st.markdown("<p style='text-align: center; color: #a0a0a0; font-size: 18px;'>Ứng dụng Trí Tuệ Nhân Tạo (CatBoost) hỗ trợ chiến lược chăm sóc khách hàng</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# CHIA BỐ CỤC 3 CỘT
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.subheader("👤 Thông tin cơ bản")
    ui_gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
    ui_senior = st.selectbox("Khách hàng cao tuổi?", ["Không", "Có"])
    
    # Ẩn 2 thông tin Vợ chồng / Người phụ thuộc vào một ô mở rộng để cho gọn
    with st.expander("Gia đình & Người phụ thuộc", expanded=False):
        st.markdown("<small style='color:gray;'>*Mô hình bắt buộc phải có các thông tin này vì đã được học từ chúng.</small>", unsafe_allow_html=True)
        ui_partner = st.selectbox("Đã lập gia đình?", ["Có", "Không"])
        ui_dependents = st.selectbox("Có người phụ thuộc (Con cái)?", ["Có", "Không"])
        
    ui_tenure = st.slider("Số tháng đã gắn bó (Tenure)", min_value=0, max_value=72, value=12)

with col2:
    st.subheader("🌐 Dịch vụ sử dụng")
    ui_phone = st.selectbox("Đăng ký số điện thoại bàn?", ["Có", "Không"])
    ui_multiple = st.selectbox("Nhiều đường dây ĐT?", ["Không", "Có", "Không dùng ĐT bàn"])
    ui_internet = st.selectbox("Gói Internet", ["Cáp quang (Fiber optic)", "Cáp đồng (DSL)", "Không dùng mạng"])
    ui_security = st.selectbox("Bảo mật mạng", ["Không", "Có", "Không dùng mạng"])
    ui_backup = st.selectbox("Lưu trữ đám mây", ["Không", "Có", "Không dùng mạng"])
    ui_protect = st.selectbox("Bảo vệ thiết bị", ["Không", "Có", "Không dùng mạng"])
    ui_support = st.selectbox("Hỗ trợ kỹ thuật VIP", ["Không", "Có", "Không dùng mạng"])
    ui_tv = st.selectbox("Truyền hình Streaming", ["Không", "Có", "Không dùng mạng"])
    ui_movies = st.selectbox("Xem phim trực tuyến", ["Không", "Có", "Không dùng mạng"])

with col3:
    st.subheader("💳 Hợp đồng & Cước phí")
    ui_contract = st.selectbox("Loại hợp đồng", ["Trả từng tháng", "Ký 1 năm", "Ký 2 năm"])
    ui_paperless = st.selectbox("Dùng Hóa đơn điện tử?", ["Có", "Không"])
    ui_payment = st.selectbox("Hình thức thanh toán", list(map_payment.keys()))
    ui_monthly = st.number_input("Cước phí hàng tháng ($)", min_value=0.0, max_value=200.0, value=50.0)
    ui_total = st.number_input("Tổng cước đã thanh toán ($)", min_value=0.0, max_value=10000.0, value=600.0)

st.markdown("<br>", unsafe_allow_html=True)

# NÚT BẤM DỰ ĐOÁN
submit = st.button("🚀 KÍCH HOẠT AI - PHÂN TÍCH RỦI RO", use_container_width=True)

if submit:
    # 1. BIẾN ĐỔI GIAO DIỆN TIẾNG VIỆT THÀNH DATA TIẾNG ANH CHO MÔ HÌNH
    input_data = pd.DataFrame({
        'gender': [map_gender[ui_gender]],
        'SeniorCitizen': [map_senior[ui_senior]],
        'Partner': [map_yes_no[ui_partner]],
        'Dependents': [map_yes_no[ui_dependents]],
        'tenure': [ui_tenure],
        'PhoneService': [map_yes_no[ui_phone]],
        'MultipleLines': [map_multiple_lines[ui_multiple]],
        'InternetService': [map_internet[ui_internet]],
        'OnlineSecurity': [map_service[ui_security]],
        'OnlineBackup': [map_service[ui_backup]],
        'DeviceProtection': [map_service[ui_protect]],
        'TechSupport': [map_service[ui_support]],
        'StreamingTV': [map_service[ui_tv]],
        'StreamingMovies': [map_service[ui_movies]],
        'Contract': [map_contract[ui_contract]],
        'PaperlessBilling': [map_yes_no[ui_paperless]],
        'PaymentMethod': [map_payment[ui_payment]],
        'MonthlyCharges': [ui_monthly],
        'TotalCharges': [ui_total]
    })
    
    with st.spinner("Mạng nơ-ron đang phân tích dữ liệu..."):
        # 2. XỬ LÝ VÀ DỰ ĐOÁN
        processed_data = preprocessor.transform(input_data)
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1] * 100
        
        # 3. HIỂN THỊ KẾT QUẢ ĐẸP MẮT
        st.markdown("---")
        st.markdown("### 📊 KẾT QUẢ ĐÁNH GIÁ TỪ HỆ THỐNG")
        
        if prediction == 1:
            st.error(f"🚨 **CẢNH BÁO MỨC ĐỎ:** Khách hàng này có **{probability:.1f}%** nguy cơ HỦY DỊCH VỤ trong tháng tới!")
            st.info("💡 **Hành động đề xuất cho Telesale:** \n- Hệ thống phát hiện rủi ro cao từ cước phí hoặc loại hợp đồng.\n- Lập tức gọi điện CSKH, đề xuất tặng Voucher giảm giá tháng đầu nếu khách hàng đồng ý chuyển sang ký Hợp đồng 1 năm.")
        else:
            st.success(f"✅ **KHÁCH HÀNG TRUNG THÀNH:** Rủi ro rời bỏ rất thấp, chỉ ở mức **{probability:.1f}%**.")
            st.info("💡 **Hành động đề xuất cho Telesale:** \n- Trạng thái an toàn.\n- Đội Sale có thể thực hiện Upsell (Bán chéo) bằng cách mời dùng thử miễn phí 1 tháng Dịch vụ Truyền hình hoặc Lưu trữ Đám mây để tăng doanh thu (LTV).")
