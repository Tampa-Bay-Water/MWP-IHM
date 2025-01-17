param(
    $fname = "Z:\MonteCarlo_Project\UCF_Landuse\MCINTB1_Q2017_LU2010_keep\mc010004\Head_Data_MonthAvg_20110101_20301231_L1_L3.BIN",
    [int]$hsdays=0,
    [int[]]$layers=@(1..3),
    [int[]]$tsteps=@(),
    [int[]]$cells=@()
)

# sort layers and tsteps
$layers = Sort-Object -InputObject $layers
$tsteps = Sort-Object -InputObject $tsteps

# INTB specific
$nwords = 4;
$nrows = 207;
$ncols = 183;
$nlays = 3;
$hbytes = 44;
$ncells = $nrows*$ncols
$nbytes = $ncells*$nwords+$hbytes #one layer

$bytearr = [System.IO.File]::ReadAllBytes($fname)
$len_arr = $bytearr.Length
$max_tstep = $len_arr/$nbytes/$nlays
if ($tsteps.Length -eq 0) { $tsteps = @(1..$max_tstep) }

# mapping cellid positions in 1-D array
if ($cells.Length -gt 0) {
    $cellord = 1..$nrows |%{
        $i = $_
        1..$ncols |%{$i*1000+$_}
    }
  $elems = $cells |%{[Array]::IndexOf($cellord,$_)}
}

# loop to read
$hmat = New-Object 'object[][]' $tsteps.Length,$layers.Length
$temp = New-Object 'Single[]' $ncells
foreach ($i in 0..($tsteps.Length-1)) {
    $t = $tsteps[$i]-1;
    foreach ($j in 0..($layers.Length-1)) {
        $l = $layers[$j]-1;
        $offset = (($hsdays+$t)*$nlays+$l)*$nbytes+$hbytes;
        if ($offset -gt $len_arr) {
            if ($debug) { Write-Warning "Timestep parameter bigger than file contents!" }
            break
        }
        [System.Buffer]::BlockCopy($bytearr, $offset, $temp, 0, $ncells*$nwords)
        if ($cells.Length -eq 0) {
            # read head for all cells in a layer 
            $hmat[$i][$j] = $temp
        }
        else {
            # read head by list of cellids
            $hmat[$i][$j] = $temp[$elems]
        }
    }
    if ($offset -gt $len_arr) {
        if ($debug) {
            Write-Warning 'Found the end of file while reading at timestep {0}' -f $t+1
        }
        break;
    }
}

return $hmat
