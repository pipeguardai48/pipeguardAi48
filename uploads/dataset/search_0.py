# -*- coding: utf-8 -*-
import os

def modify_class_in_labels(labels_dir, old_class=0, new_class=8, save_dir=None):
    """
    يعدل الكلاس في ملفات YOLO label من قيمة معينة (مثلاً 0) إلى قيمة جديدة (مثلاً 8)
    """
    if save_dir is None:
        save_dir = labels_dir  # حفظ في نفس المجلد إذا لم يُحدد مجلد جديد

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    count_modified = 0

    for file in os.listdir(labels_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(labels_dir, file)
            with open(file_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            modified = False

            for line in lines:
                parts = line.strip().split(" ")
                if len(parts) > 0 and parts[0] == str(old_class):
                    parts[0] = str(new_class)
                    modified = True
                new_lines.append(" ".join(parts) + "\n")

            if modified:
                count_modified += 1

            new_path = os.path.join(save_dir, file)
            with open(new_path, "w") as f:
                f.writelines(new_lines)

    print("✅ Done! Modified {} label files ({} → {}).".format(count_modified, old_class, new_class))


# 🔹 مثال التشغيل
labels_dir = "valid/labels"   # ✅ غيّر حسب مسارك
modify_class_in_labels(labels_dir, old_class=0, new_class=8)
