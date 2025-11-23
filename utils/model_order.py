def model_order(data):
    model_order_list = [
        "YOLOv8n",
        "YOLOv8s",
        "YOLOv8m",
        "YOLOv9t",
        "YOLOv9s",
        "YOLOv9m",
        "YOLOv10n",
        "YOLOv10s",
        "YOLOv10m",
        "YOLOv11n",
        "YOLOv11s",
        "YOLOv11m",
        "YOLOv12n",
        "YOLOv12s",
        "YOLOv12m",
    ]

    # Tambah kolom Order sesuai urutan model
    data["Order"] = data["Model"].apply(
        lambda x: model_order_list.index(x) + 1 if x in model_order_list else 999
    )

    data = data.sort_values("Order").reset_index(drop=True)

    # Tambah kolom Index dimulai dari 1
    data.insert(0, "Index", range(1, len(data) + 1))

    data = data.drop(columns=["Order"])

    return data
