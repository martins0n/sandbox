NAME="ffnet-electricity"

report:
	@ echo  "## ${NAME}" > ${NAME}.md
	@ echo  "![${NAME}](assets/${NAME}.png)" >> ${NAME}.md
	@ cat assets/${NAME}.metrics >> ${NAME}.md

format:
	@ isort ./*.py
	@ black ./*.py