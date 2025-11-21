import os
import shutil

base_path = r"E:\\high_DiskSpy\\img\\disk_8_75ms_2"

def copy_with_suffix(src_dir, dst_dir, suffix):
    """复制并在文件名加上后缀"""
    if not os.path.exists(src_dir):
        print(f"⚠️ 跳过不存在的目录：{src_dir}")
        return
    os.makedirs(dst_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)
        if not os.path.isfile(src_path):
            continue  # 跳过文件夹

        name, ext = os.path.splitext(fname)
        new_name = f"{name}{suffix}{ext}"
        dst_path = os.path.join(dst_dir, new_name)

        shutil.copy2(src_path, dst_path)
        # print(f"✅ Copied: {src_path} → {dst_path}")

def main():
    for x in range(4):  # x = 0, 1, 2, 3
        db_src = os.path.join(base_path, "db", "cut", str(x))
        # amp_src = os.path.join(base_path, "amp", "cut", str(x))
        # db_src = os.path.join(base_path, "db", str(x))
        # amp_src = os.path.join(base_path, "amp", str(x))
        dst_dir = os.path.join(base_path, "dataset", str(x))

        print(f"\n--- 处理目录 x={x} ---")
        copy_with_suffix(db_src, dst_dir, "_db")
        # copy_with_suffix(amp_src, dst_dir, "_amp")

    print("\n🎉 所有目录处理完成！")

if __name__ == "__main__":
    main()
