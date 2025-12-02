import os
import numpy as np
import mne

# --- CẤU HÌNH ĐƯỜNG DẪN CỦA BẠN ---
# Đường dẫn đến thư mục chứa data trên máy bạn
DATA_DIR = r"C:\NCKH2025\Code PyCharm\NCKH\Data"

# Tên file bạn đang có
PSG_FILE = os.path.join(DATA_DIR, "SC4001E0-PSG.edf")
HYP_FILE = os.path.join(DATA_DIR, "SC4001EC-Hypnogram.edf")

# Nơi sẽ lưu file sample (để up lên GitHub)
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sample_subject.npz")

def create_sample_from_edf():
    print(f"🔄 Đang đọc dữ liệu từ: {PSG_FILE}")
    
    # 1. Kiểm tra file tồn tại
    if not os.path.exists(PSG_FILE) or not os.path.exists(HYP_FILE):
        print(f"❌ Lỗi: Không tìm thấy file EDF tại {DATA_DIR}")
        return

    # 2. Load dữ liệu Raw bằng MNE
    # preload=True để load vào RAM xử lý cho nhanh
    raw = mne.io.read_raw_edf(PSG_FILE, preload=True, verbose=False)
    annot = mne.read_annotations(HYP_FILE)
    
    # Gắn nhãn vào tín hiệu
    raw.set_annotations(annot, emit_warning=False)

    # 3. Chọn kênh tín hiệu quan trọng (Theo chuẩn Sleep-EDF)
    # Thường là EEG Fpz-Cz và Pz-Oz. Nếu code train của bạn dùng kênh khác, hãy sửa list này.
    include_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
    try:
        raw.pick_channels(include_channels)
    except ValueError:
        print(f"⚠️ Cảnh báo: Không tìm thấy kênh chính xác. Các kênh hiện có: {raw.ch_names}")
        # Lấy 2 kênh đầu tiên làm mẫu nếu không tìm thấy tên đúng
        raw.pick_channels(raw.ch_names[:2])

    print(f"✅ Đã chọn kênh: {raw.ch_names}")

    # 4. Cắt lấy dữ liệu mẫu (Lấy 60 phút đầu tiên = 3600 giây)
    # File gốc rất dài, ta chỉ lấy 1 đoạn để làm mẫu
    DURATION_SEC = 3600 
    raw_cropped = raw.crop(tmin=0, tmax=DURATION_SEC)
    
    # 5. Xử lý Epochs (Cắt thành các đoạn 30 giây)
    # Sleep Staging tiêu chuẩn dùng cửa sổ 30s
    EPOCH_DURATION = 30.
    
    # Lấy sự kiện từ annotation để cắt đúng nhãn
    events, event_id = mne.events_from_annotations(
        raw_cropped, 
        event_id={'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 
                  'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4},
        chunk_duration=EPOCH_DURATION,
        verbose=False
    )
    
    # Tạo Epochs
    epochs = mne.Epochs(
        raw_cropped, 
        events, 
        event_id=event_id, 
        tmin=0, 
        tmax=EPOCH_DURATION - 1/raw.info['sfreq'], 
        baseline=None,
        verbose=False
    )

    # 6. Chuyển sang dạng Numpy Array
    # X shape: (Số mẫu, Số kênh, Số điểm ảnh) hoặc (Số mẫu, Số điểm ảnh, Số kênh) tùy code train
    # Ở đây mình để dạng chuẩn (N_epochs, N_times, N_channels) -> Transpose nếu cần
    X = epochs.get_data() # (N_epochs, N_channels, N_times)
    y = epochs.events[:, 2] # Lấy cột nhãn
    
    # Chuyển axis để khớp với code Keras thường dùng: (Batch, Time, Channel)
    X = np.moveaxis(X, 1, 2) 

    print(f"📊 Kích thước dữ liệu mẫu: X={X.shape}, y={y.shape}")

    # 7. Lưu file .npz
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    np.savez(OUTPUT_FILE, x=X, y=y, fs=raw.info['sfreq'], ch_names=raw.ch_names)
    
    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"✅ Đã tạo file thành công: {OUTPUT_FILE}")
    print(f"📉 Dung lượng: {file_size:.2f} MB (An toàn để up lên GitHub)")

if __name__ == "__main__":
    # Cài thư viện nếu chưa có: pip install mne numpy
    create_sample_from_edf()