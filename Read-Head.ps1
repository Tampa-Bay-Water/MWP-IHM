# function Read-Head: read headfile by layers and/or by CellIDs
param (
	[String]$fn,			# fn = path to head file as string                      
	[int]$hsdays=0,			# hsdays =  number of hotstart in days (tsteps)         
	[int[]]$layers=@(1,3),	# layers = the vector which elements are layers to read 
	[int[]]$tsteps=@(1..7),	# tsteps = the vector which elements are tsteps to read 
	[int[]]$cells=@(),		# cells = the vector which elements are cellid's to read
                 			# (CellIDs need NOT to be sorted)
	[switch]$debug
)
# hmat = return 3-D matrix of heads read from the read file with these
#        dimensions: length(cells) X length(tsteps) X length(layers).
#        Return heads are ordered by cellid, time step and layer along
#        each respective dimension.
trap {
	Write-Host $error[0] -fore Red
	exit;
}
$error.Clear()

#-------------------------------------------------------------------------------
function Read-HeadParams {
	param ([System.IO.BinaryReader]$binReader)
	# get headfile parameters
	# header for each time step is 44 bytes of the following information
	# TimeStep As Integer (4)
	# Period As Integer (4)
	# PeriodTime As Single (4)
	# TotalTime As Single (4)
	# Text As String 16 bytes
	# ByRef Columns As Integer (4)
	# ByRef Rows As Integer (4)
	# ByRef Layers As Integer (4)
	$BinType = $binReader.ReadInt32();
	$Period = $binReader.ReadInt32();
	$PeriodTime = $binReader.ReadSingle();
	$TotalTime = $binReader.ReadSingle();
	$Text = $binReader.ReadChars(16);
	$Columns = $binReader.ReadInt32();
	$Rows = $binReader.ReadInt32();
	$Layer = $binReader.ReadInt32();
	@($BinType,$Period,$PeriodTime,$TotalTime,$Text,$Columns,$Rows,$Layer)
}
#-------------------------------------------------------------------------------

$nwords = 4;
$nrows = 207;
$ncols = 183;
$nlays = 3;
$hbytes = 44;
$nbytes = $nrows*$ncols*$nwords+$hbytes; # one layer
$cellord = 1..$nrows |%{ $i = $_; 1..$ncols |%{$i*1000+$_} } 

try {
	$fid = New-Object System.IO.FileStream($fn,[System.IO.FileMode]::Open)
}
catch {
	[System.Exception]"ReadHead: Can't open '$fn'!"
}
$br = New-Object System.IO.BinaryReader($fid)

# sort layers and tsteps
$layers = $layers |sort
$tsteps = $tsteps |sort

# determine cell positions as element numbers
$elems = @()
if ($cells.Count > 0) {
	$elems = 0..($cellord.Count-1) |where {$cells -contains $cellord[$_]}
}

# loop to read
$hmat = @()
foreach ($i in (0..($tsteps.Count-1))) {
	$t = $tsteps[$i]-1
	foreach ($j in (0..($layers.Count-1))) {
		$l = $layers[$j]-1
		$offset = (($hsdays+$t)*$nlays+$l)*$nbytes
		if (-not $debug) { $offset += $hbytes }
		$temp = $br.BaseStream.Seek($offset, [System.IO.SeekOrigin]::Begin)
		if ($debug) { $bin_pars = Read-HeadParams($br) }
		
		$temp = 0..($nrows*$ncols-1) |%{$br.ReadSingle()}
		if ($cells.Count -gt 0) { $hmat += $temp[$elems] }	# read head by list of cellids
		else { $hmat += $temp } # read head for all cells in a layer 

	}
}

$br.Close()
$br.Dispose()
$fid.Close()
$fid.Dispose()

$retval = @{}
if ($cells.Count -gt 0) {
    # support 1 cell in URM - reduce to 2-D matrix
    if ($cells.Count -eq 1) {
		foreach ($l in (0..($layers.Count-1))) {
        	$retval[$layers[$l].ToString('Layer0')] = 
				$hmat[($l*$tsteps.Count)..(($l+1)*$tsteps.Count-1)]
		}
    }
	else {
		$nt = $cells.Count*$tsteps.Count
		foreach ($l in (0..($layers.Count-1))) {
			$retval[$layers[$l].ToString('Layer0')] = @{}
			foreach ($t in (0..($tsteps.Count-1))) {
        		$retval[$layers[$l].ToString('Layer0')][$tsteps[$t]] = 
					$hmat[(($t*$cells.Count)+$nt*$l)..((($t+1)*$cells.Count-1)+$nt*$l)]
			}
		}
	}
}
else {
	$nc = $nrows*$ncols
    # support 1 tstep in URM - reduce to 2-D matrix
	if ($tsteps.Count -eq 1) {
		foreach ($l in (0..($layers.Count-1))) {
			$retval[$layers[$l].ToString('Layer0')] = $hmat[($nc*$l)..($nc*($l+1)-1)]
		}
	}
	elseif ($layers.Count -eq 1) {
		foreach ($t in (0..($tsteps.Count-1))) {
			$retval[$tsteps[$t]] = $hmat[($nc*$t)..($nc*($t+1)-1)]
		}
	}
    else {
		$nt = $nc*$tsteps.Count
		foreach ($l in (0..($layers.Count-1))) {
			$retval[$layers[$l].ToString('Layer0')] = @{}
			foreach ($t in (0..($tsteps.Count-1))) {
        		$retval[$layers[$l].ToString('Layer0')][$tsteps[$t]] = 
					$hmat[(($t*$nc)+$nt*$l)..((($t+1)*$nc-1)+$nt*$l)]
			}
		}
	}
}
return $retval
