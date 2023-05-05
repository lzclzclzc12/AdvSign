(
out_name = "D:/PyCharmProject/Full-coverage-camouflage-adversarial-attack-gh-pages/src/carassets/new_mm/new_mm_faces.txt"
out_file = createfile out_name

fSel = $.selectedFaces	
for face in (fSel) do (
	format "%\n" face.index to:out_file
	)
					
close out_file
)