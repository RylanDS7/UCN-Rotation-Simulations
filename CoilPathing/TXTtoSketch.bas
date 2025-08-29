Attribute VB_Name = "CSVtoSketch1"
Dim swApp As SldWorks.SldWorks
Dim swModel As ModelDoc2
Dim swSketchMgr As SketchManager
Dim swSketchSeg As SketchSegment
Dim swPoint As SketchPoint
Dim filePath As String
Dim line As String
Dim splitLine() As String
Dim fileNum As Integer
Dim PointArray() As Double
Dim pointCount As Long
Dim i As Integer

Sub sketch_contour(Optional ByVal argFilePath As String = "")

    ' Set the active SolidWorks application and model
    Set swApp = Application.SldWorks
    Set swModel = swApp.ActiveDoc
    
    ' Ensure there is an open part document
    If swModel Is Nothing Or swModel.GetType <> swDocPART Then
        MsgBox "Please open a part document to run this macro."
        Exit Sub
    End If

    ' Set the path to the file
    filePath = argFilePath

    ' Read and parse the file
    fileNum = FreeFile
    Open filePath For Input As #fileNum

    ' Begin a new 3D sketch
    Set swSketchMgr = swModel.SketchManager
    swSketchMgr.Insert3DSketch True

    ' Read each line and extract the coordinates to create a point
    pointCount = 0
    Do While Not EOF(fileNum)
        Line Input #fileNum, line
        splitLine = Split(line, ",")
        Dim x As Double, y As Double, z As Double
        x = CDbl(splitLine(0)) / 1000
        y = CDbl(splitLine(1)) / 1000
        z = CDbl(splitLine(2)) / 1000
        swSketchMgr.CreatePoint x, y, z
        
        ' Add the coordinates to the point array
        ReDim Preserve PointArray(pointCount * 3 + 2)
        PointArray(pointCount * 3) = x
        PointArray(pointCount * 3 + 1) = y
        PointArray(pointCount * 3 + 2) = z
        pointCount = pointCount + 1
    Loop

    ' Close the CSV file
    Close #fileNum
    
    If pointCount > 1 Then
        Set swSketchSeg = swSketchMgr.CreateSpline(PointArray)
    End If

    ' End the 3D sketch
    swSketchMgr.Insert3DSketch False
    bRet = swModel.ForceRebuild3(False)

End Sub

Sub main()

i = 1
Do While i < 6
    sketch_contour ("C:\Users\rstutters\Desktop\Test Coil 1\Contours\TestCoil-C" & CStr(i) & ".txt")
    i = i + 1
Loop

End Sub
