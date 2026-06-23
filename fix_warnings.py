with open("frontend/app.py", "r") as f:
    content = f.read()

content = content.replace("use_container_width=True", 'width="stretch"')

with open("frontend/app.py", "w") as f:
    f.write(content)

print("Warnings fixed")
