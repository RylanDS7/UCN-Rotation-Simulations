# Save cleaned contours
os.mkdir("CoilPathing\Contours_out" + input_file)
i = 0
for contour in contours:
    contour.to_csv("CoilPathing\Contours_out" + input_file + f"\C-{i}", index=False, sep=",", float_format="%f", header=False)
    i += 1