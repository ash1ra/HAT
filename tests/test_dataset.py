from pathlib import Path
from datasets import StaticPairDataset, DynamicPairDataset
import pytest
from PIL import Image


DATA_CONFIG = [
    # scaling_factor, patch_size
    (2, 16),
    (4, 32),
    (8, 8),
    (3, 16),
]


@pytest.fixture
def fake_root_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


def create_fake_data_static(
    root: Path,
    scaling_factor: int,
    num_imgs: int = 10,
    imgs_size: tuple[int, int] = (200, 200),
) -> None:
    hr_dir = root / "HR"
    lr_dir = root / f"LR_x{scaling_factor}"

    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_imgs):
        Image.new("RGB", imgs_size, color="red").save(hr_dir / f"img_{i}.png")
        Image.new("RGB", (imgs_size[0] // scaling_factor, imgs_size[1] // scaling_factor), color="blue").save(
            lr_dir / f"img_{i}.png"
        )


def create_fake_data_dynamic(
    root: Path,
    num_imgs: int = 10,
    imgs_size: tuple[int, int] = (200, 200),
) -> Path:
    img_dir = root / "Dynamic"
    img_dir.mkdir(parents=True, exist_ok=True)

    list_file_path = root / "train_list.txt"

    paths = []
    for i in range(num_imgs):
        path = img_dir / f"img_{i}.png"
        Image.new("RGB", imgs_size, color="green").save(path)
        paths.append(str(path.absolute()))

    with open(list_file_path, "w") as f:
        f.write("\n".join(paths))

    return list_file_path


@pytest.mark.parametrize("scaling_factor, patch_size", DATA_CONFIG)
def test_static_dataset_initialization(fake_root_dir: Path, scaling_factor: int, patch_size: int) -> None:
    create_fake_data_static(root=fake_root_dir, scaling_factor=scaling_factor)

    Image.new("RGB", (100, 100)).save(fake_root_dir / "HR" / "hr_orphan.png")
    Image.new("RGB", (25, 25)).save(fake_root_dir / f"LR_x{scaling_factor}" / "lr_orphan.png")

    dataset = StaticPairDataset(
        data_path=fake_root_dir,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
    )

    assert len(dataset) == 10


@pytest.mark.parametrize("scaling_factor, patch_size", DATA_CONFIG)
def test_static_dataset_output_shape(fake_root_dir: Path, scaling_factor: int, patch_size: int) -> None:
    create_fake_data_static(root=fake_root_dir, scaling_factor=scaling_factor)

    dataset = StaticPairDataset(
        data_path=fake_root_dir,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
    )

    imgs_tensor = dataset[0]
    lr_img_tensor = imgs_tensor["lr"]
    hr_img_tensor = imgs_tensor["hr"]

    assert lr_img_tensor.shape == (3, patch_size, patch_size)
    assert hr_img_tensor.shape == (3, patch_size * scaling_factor, patch_size * scaling_factor)


@pytest.mark.parametrize("scaling_factor, patch_size", DATA_CONFIG)
def test_static_dataset_normalization(fake_root_dir: Path, scaling_factor: int, patch_size: int) -> None:
    create_fake_data_static(root=fake_root_dir, scaling_factor=scaling_factor)

    dataset = StaticPairDataset(
        data_path=fake_root_dir,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
    )

    imgs_tensor = dataset[0]

    assert imgs_tensor["hr"].max() <= 1.0
    assert imgs_tensor["hr"].min() >= 0.0

    assert imgs_tensor["lr"].max() <= 1.0
    assert imgs_tensor["lr"].min() >= 0.0


@pytest.mark.parametrize("scaling_factor, patch_size", DATA_CONFIG)
def test_static_dataset_small_image(fake_root_dir: Path, scaling_factor: int, patch_size: int) -> None:
    create_fake_data_static(root=fake_root_dir, scaling_factor=scaling_factor)

    Image.new("RGB", (5, 5)).save(fake_root_dir / "HR" / "tiny.png")
    Image.new("RGB", (5, 5)).save(fake_root_dir / f"LR_x{scaling_factor}" / "tiny.png")

    dataset = StaticPairDataset(
        data_path=fake_root_dir,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
    )

    for i in range(len(dataset)):
        imgs_tensor = dataset[i]
        assert imgs_tensor["lr"].shape == (3, patch_size, patch_size)


@pytest.mark.parametrize("scaling_factor, patch_size", DATA_CONFIG)
def test_static_dataset_augmentations(fake_root_dir: Path, scaling_factor: int, patch_size: int) -> None:
    create_fake_data_static(root=fake_root_dir, scaling_factor=scaling_factor)

    dataset = StaticPairDataset(
        data_path=fake_root_dir,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
    )

    for _ in range(50):
        imgs_tensor = dataset[0]
        assert imgs_tensor["hr"].shape == (3, patch_size * scaling_factor, patch_size * scaling_factor)


@pytest.mark.parametrize("scaling_factor, patch_size", DATA_CONFIG)
def test_static_dataset_test_mode(fake_root_dir: Path, scaling_factor: int, patch_size: int) -> None:
    create_fake_data_static(root=fake_root_dir, scaling_factor=scaling_factor)

    img_height = 200
    img_width = 200

    dataset = StaticPairDataset(
        data_path=fake_root_dir,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
        test_mode=True,
    )

    imgs_tensor = dataset[0]

    assert imgs_tensor["hr"].shape == (3, img_height, img_width)
    assert imgs_tensor["lr"].shape == (3, img_height // scaling_factor, img_width // scaling_factor)


@pytest.mark.parametrize("scaling_factor, patch_size", DATA_CONFIG)
def test_static_dataset_dev_mode(fake_root_dir: Path, scaling_factor: int, patch_size: int) -> None:
    create_fake_data_static(root=fake_root_dir, scaling_factor=scaling_factor)

    dataset = StaticPairDataset(
        data_path=fake_root_dir,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
        dev_mode=True,
    )

    assert len(dataset) == 1


@pytest.mark.parametrize("scaling_factor, patch_size", DATA_CONFIG)
def test_dynamic_dataset_initialization(fake_root_dir: Path, scaling_factor: int, patch_size: int) -> None:
    list_file = create_fake_data_dynamic(root=fake_root_dir)

    dataset = DynamicPairDataset(
        data_path=list_file,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
    )

    assert len(dataset) == 10


@pytest.mark.parametrize("scaling_factor, patch_size", DATA_CONFIG)
def test_dynamic_dataset_output_shape(fake_root_dir: Path, scaling_factor: int, patch_size: int) -> None:
    list_file = create_fake_data_dynamic(root=fake_root_dir)

    dataset = DynamicPairDataset(
        data_path=list_file,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
    )

    imgs_tensor = dataset[0]
    lr_img_tensor = imgs_tensor["lr"]
    hr_img_tensor = imgs_tensor["hr"]

    assert lr_img_tensor.shape == (3, patch_size, patch_size)
    assert hr_img_tensor.shape == (3, patch_size * scaling_factor, patch_size * scaling_factor)


@pytest.mark.parametrize("scaling_factor, patch_size", DATA_CONFIG)
def test_dynamic_dataset_normalization(fake_root_dir: Path, scaling_factor: int, patch_size: int) -> None:
    list_file = create_fake_data_dynamic(root=fake_root_dir)

    dataset = DynamicPairDataset(
        data_path=list_file,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
    )

    imgs_tensor = dataset[0]

    assert imgs_tensor["hr"].max() <= 1.0
    assert imgs_tensor["hr"].min() >= 0.0

    assert imgs_tensor["lr"].max() <= 1.0
    assert imgs_tensor["lr"].min() >= 0.0


@pytest.mark.parametrize("scaling_factor, patch_size", DATA_CONFIG)
def test_dynamic_dataset_small_image(fake_root_dir: Path, scaling_factor: int, patch_size: int) -> None:
    list_file = create_fake_data_dynamic(root=fake_root_dir)

    Image.new("RGB", (5, 5)).save(fake_root_dir / "Dynamic" / "tiny.png")

    dataset = DynamicPairDataset(
        data_path=list_file,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
    )

    for i in range(len(dataset)):
        imgs_tensor = dataset[i]
        assert imgs_tensor["lr"].shape == (3, patch_size, patch_size)


@pytest.mark.parametrize("scaling_factor, patch_size", DATA_CONFIG)
def test_dynamic_dataset_augmentations(fake_root_dir: Path, scaling_factor: int, patch_size: int) -> None:
    list_file = create_fake_data_dynamic(root=fake_root_dir)

    dataset = DynamicPairDataset(
        data_path=list_file,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
    )

    for _ in range(50):
        imgs_tensor = dataset[0]
        assert imgs_tensor["hr"].shape == (3, patch_size * scaling_factor, patch_size * scaling_factor)


@pytest.mark.parametrize("scaling_factor, patch_size", DATA_CONFIG)
def test_dynamic_dataset_test_mode(fake_root_dir: Path, scaling_factor: int, patch_size: int) -> None:
    list_file = create_fake_data_dynamic(root=fake_root_dir)

    dataset = DynamicPairDataset(
        data_path=list_file,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
        test_mode=True,
    )

    img_height = 200
    img_width = 200

    expected_img_height = img_height - img_height % scaling_factor
    expected_img_width = img_width - img_width % scaling_factor

    imgs_tensor = dataset[0]

    assert imgs_tensor["hr"].shape == (3, expected_img_height, expected_img_width)
    assert imgs_tensor["lr"].shape == (3, expected_img_height // scaling_factor, expected_img_width // scaling_factor)


@pytest.mark.parametrize("scaling_factor, patch_size", DATA_CONFIG)
def test_dynamic_dataset_dev_mode(fake_root_dir: Path, scaling_factor: int, patch_size: int) -> None:
    list_file = create_fake_data_dynamic(root=fake_root_dir)

    dataset = DynamicPairDataset(
        data_path=list_file,
        scaling_factor=scaling_factor,
        patch_size=patch_size,
        dev_mode=True,
    )

    assert len(dataset) == 1
