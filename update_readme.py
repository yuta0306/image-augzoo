import os
from typing import Dict, List


def find_all() -> Dict[str, List[str]]:
    data = {}
    for cdir, _, images in os.walk("assets"):
        data[cdir] = images
    return data


def generate_markdown(data: Dict[str, List[str]]) -> str:
    # github actions badges
    string = (
        "[![python3.7](https://github.com/yuta0306/image-augzoo/actions/workflows/ci-py37.yml/badge.svg)]"
        "(https://github.com/yuta0306/image-augzoo/actions/workflows/ci-py37.yml)"
        "[![python3.8](https://github.com/yuta0306/image-augzoo/actions/workflows/ci-py38.yml/badge.svg)]"
        "(https://github.com/yuta0306/image-augzoo/actions/workflows/ci-py38.yml)"
        "[![python3.9](https://github.com/yuta0306/image-augzoo/actions/workflows/ci-py39.yml/badge.svg)]"
        "(https://github.com/yuta0306/image-augzoo/actions/workflows/ci-py39.yml)"
        "[![python3.10](https://github.com/yuta0306/image-augzoo/actions/workflows/ci-py310.yml/badge.svg)]"
        "(https://github.com/yuta0306/image-augzoo/actions/workflows/ci-py310.yml)"
    )
    string += "\n\n"
    # title
    string += "# Image Augzoo\n\n"
    # description
    string += "Image Augmentation Zoo\n\n"
    # images
    for method, images in data.items():
        cdir = method
        method = method.replace("assets", "").replace("/", "")
        if method == "":
            method = "original"
        string += f"## {method}\n\n"
        for image in images:
            # subtitle
            if method == "original":
                string += f"![{image}](https://github.com/yuta0306/image-augzoo/{cdir}/{image})"
            else:
                subtitle = image.replace(".png", "").replace(".jpg", "")
                string += f"### {subtitle}\n\n"
                string += f"![{subtitle}](https://github.com/yuta0306/image-augzoo/{cdir}/{image})\n\n"
        if method == "original":
            string += "\n\n"

    return string


def write_markdown(content: str) -> None:
    with open("README.md", "w") as f:
        f.write(content)


if __name__ == "__main__":
    data = find_all()
    readme = generate_markdown(data=data)
    write_markdown(readme)
