¶£
Â¦
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02unknown8Èì

extract_features/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameextract_features/kernel

+extract_features/kernel/Read/ReadVariableOpReadVariableOpextract_features/kernel*"
_output_shapes
:@*
dtype0

extract_features/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameextract_features/bias
{
)extract_features/bias/Read/ReadVariableOpReadVariableOpextract_features/bias*
_output_shapes
:@*
dtype0

extract_features_BN/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_nameextract_features_BN/gamma

-extract_features_BN/gamma/Read/ReadVariableOpReadVariableOpextract_features_BN/gamma*
_output_shapes
:@*
dtype0

extract_features_BN/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameextract_features_BN/beta

,extract_features_BN/beta/Read/ReadVariableOpReadVariableOpextract_features_BN/beta*
_output_shapes
:@*
dtype0

extract_features_BN/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!extract_features_BN/moving_mean

3extract_features_BN/moving_mean/Read/ReadVariableOpReadVariableOpextract_features_BN/moving_mean*
_output_shapes
:@*
dtype0

#extract_features_BN/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#extract_features_BN/moving_variance

7extract_features_BN/moving_variance/Read/ReadVariableOpReadVariableOp#extract_features_BN/moving_variance*
_output_shapes
:@*
dtype0
x
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv1/kernel
q
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*"
_output_shapes
:@@*
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
:@*
dtype0
t
conv1_BN/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1_BN/gamma
m
"conv1_BN/gamma/Read/ReadVariableOpReadVariableOpconv1_BN/gamma*
_output_shapes
:@*
dtype0
r
conv1_BN/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1_BN/beta
k
!conv1_BN/beta/Read/ReadVariableOpReadVariableOpconv1_BN/beta*
_output_shapes
:@*
dtype0

conv1_BN/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameconv1_BN/moving_mean
y
(conv1_BN/moving_mean/Read/ReadVariableOpReadVariableOpconv1_BN/moving_mean*
_output_shapes
:@*
dtype0

conv1_BN/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv1_BN/moving_variance

,conv1_BN/moving_variance/Read/ReadVariableOpReadVariableOpconv1_BN/moving_variance*
_output_shapes
:@*
dtype0
x
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv2/kernel
q
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*"
_output_shapes
:@@*
dtype0
l

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
conv2/bias
e
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes
:@*
dtype0
t
conv2_BN/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2_BN/gamma
m
"conv2_BN/gamma/Read/ReadVariableOpReadVariableOpconv2_BN/gamma*
_output_shapes
:@*
dtype0
r
conv2_BN/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2_BN/beta
k
!conv2_BN/beta/Read/ReadVariableOpReadVariableOpconv2_BN/beta*
_output_shapes
:@*
dtype0

conv2_BN/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameconv2_BN/moving_mean
y
(conv2_BN/moving_mean/Read/ReadVariableOpReadVariableOpconv2_BN/moving_mean*
_output_shapes
:@*
dtype0

conv2_BN/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameconv2_BN/moving_variance

,conv2_BN/moving_variance/Read/ReadVariableOpReadVariableOpconv2_BN/moving_variance*
_output_shapes
:@*
dtype0
{
d1_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_named1_dense/kernel
t
#d1_dense/kernel/Read/ReadVariableOpReadVariableOpd1_dense/kernel*
_output_shapes
:	@*
dtype0
s
d1_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_named1_dense/bias
l
!d1_dense/bias/Read/ReadVariableOpReadVariableOpd1_dense/bias*
_output_shapes	
:*
dtype0
o
d1_BN/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_named1_BN/gamma
h
d1_BN/gamma/Read/ReadVariableOpReadVariableOpd1_BN/gamma*
_output_shapes	
:*
dtype0
m

d1_BN/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
d1_BN/beta
f
d1_BN/beta/Read/ReadVariableOpReadVariableOp
d1_BN/beta*
_output_shapes	
:*
dtype0
{
d1_BN/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_named1_BN/moving_mean
t
%d1_BN/moving_mean/Read/ReadVariableOpReadVariableOpd1_BN/moving_mean*
_output_shapes	
:*
dtype0

d1_BN/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_named1_BN/moving_variance
|
)d1_BN/moving_variance/Read/ReadVariableOpReadVariableOpd1_BN/moving_variance*
_output_shapes	
:*
dtype0
|
d2_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_named2_dense/kernel
u
#d2_dense/kernel/Read/ReadVariableOpReadVariableOpd2_dense/kernel* 
_output_shapes
:
*
dtype0
s
d2_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_named2_dense/bias
l
!d2_dense/bias/Read/ReadVariableOpReadVariableOpd2_dense/bias*
_output_shapes	
:*
dtype0
o
d2_BN/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_named2_BN/gamma
h
d2_BN/gamma/Read/ReadVariableOpReadVariableOpd2_BN/gamma*
_output_shapes	
:*
dtype0
m

d2_BN/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
d2_BN/beta
f
d2_BN/beta/Read/ReadVariableOpReadVariableOp
d2_BN/beta*
_output_shapes	
:*
dtype0
{
d2_BN/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_named2_BN/moving_mean
t
%d2_BN/moving_mean/Read/ReadVariableOpReadVariableOpd2_BN/moving_mean*
_output_shapes	
:*
dtype0

d2_BN/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_named2_BN/moving_variance
|
)d2_BN/moving_variance/Read/ReadVariableOpReadVariableOpd2_BN/moving_variance*
_output_shapes	
:*
dtype0

dense_predict/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_namedense_predict/kernel
~
(dense_predict/kernel/Read/ReadVariableOpReadVariableOpdense_predict/kernel*
_output_shapes
:	*
dtype0
|
dense_predict/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namedense_predict/bias
u
&dense_predict/bias/Read/ReadVariableOpReadVariableOpdense_predict/bias*
_output_shapes
:*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
t
cond_1/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *!
shared_namecond_1/Adam/iter
m
$cond_1/Adam/iter/Read/ReadVariableOpReadVariableOpcond_1/Adam/iter*
_output_shapes
: *
dtype0	
x
current_loss_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_namecurrent_loss_scale
q
&current_loss_scale/Read/ReadVariableOpReadVariableOpcurrent_loss_scale*
_output_shapes
: *
dtype0
h

good_stepsVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
good_steps
a
good_steps/Read/ReadVariableOpReadVariableOp
good_steps*
_output_shapes
: *
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
`
AUCsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAUCs
Y
AUCs/Read/ReadVariableOpReadVariableOpAUCs*
_output_shapes
:*
dtype0
\
NsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameNs
U
Ns/Read/ReadVariableOpReadVariableOpNs*
_output_shapes
:*
dtype0
ª
%cond_1/Adam/extract_features/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%cond_1/Adam/extract_features/kernel/m
£
9cond_1/Adam/extract_features/kernel/m/Read/ReadVariableOpReadVariableOp%cond_1/Adam/extract_features/kernel/m*"
_output_shapes
:@*
dtype0

#cond_1/Adam/extract_features/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#cond_1/Adam/extract_features/bias/m

7cond_1/Adam/extract_features/bias/m/Read/ReadVariableOpReadVariableOp#cond_1/Adam/extract_features/bias/m*
_output_shapes
:@*
dtype0
¦
'cond_1/Adam/extract_features_BN/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'cond_1/Adam/extract_features_BN/gamma/m

;cond_1/Adam/extract_features_BN/gamma/m/Read/ReadVariableOpReadVariableOp'cond_1/Adam/extract_features_BN/gamma/m*
_output_shapes
:@*
dtype0
¤
&cond_1/Adam/extract_features_BN/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&cond_1/Adam/extract_features_BN/beta/m

:cond_1/Adam/extract_features_BN/beta/m/Read/ReadVariableOpReadVariableOp&cond_1/Adam/extract_features_BN/beta/m*
_output_shapes
:@*
dtype0

cond_1/Adam/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*+
shared_namecond_1/Adam/conv1/kernel/m

.cond_1/Adam/conv1/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1/kernel/m*"
_output_shapes
:@@*
dtype0

cond_1/Adam/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namecond_1/Adam/conv1/bias/m

,cond_1/Adam/conv1/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1/bias/m*
_output_shapes
:@*
dtype0

cond_1/Adam/conv1_BN/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecond_1/Adam/conv1_BN/gamma/m

0cond_1/Adam/conv1_BN/gamma/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1_BN/gamma/m*
_output_shapes
:@*
dtype0

cond_1/Adam/conv1_BN/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namecond_1/Adam/conv1_BN/beta/m

/cond_1/Adam/conv1_BN/beta/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1_BN/beta/m*
_output_shapes
:@*
dtype0

cond_1/Adam/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*+
shared_namecond_1/Adam/conv2/kernel/m

.cond_1/Adam/conv2/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2/kernel/m*"
_output_shapes
:@@*
dtype0

cond_1/Adam/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namecond_1/Adam/conv2/bias/m

,cond_1/Adam/conv2/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2/bias/m*
_output_shapes
:@*
dtype0

cond_1/Adam/conv2_BN/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecond_1/Adam/conv2_BN/gamma/m

0cond_1/Adam/conv2_BN/gamma/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2_BN/gamma/m*
_output_shapes
:@*
dtype0

cond_1/Adam/conv2_BN/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namecond_1/Adam/conv2_BN/beta/m

/cond_1/Adam/conv2_BN/beta/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2_BN/beta/m*
_output_shapes
:@*
dtype0

cond_1/Adam/d1_dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*.
shared_namecond_1/Adam/d1_dense/kernel/m

1cond_1/Adam/d1_dense/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/d1_dense/kernel/m*
_output_shapes
:	@*
dtype0

cond_1/Adam/d1_dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/d1_dense/bias/m

/cond_1/Adam/d1_dense/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/d1_dense/bias/m*
_output_shapes	
:*
dtype0

cond_1/Adam/d1_BN/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecond_1/Adam/d1_BN/gamma/m

-cond_1/Adam/d1_BN/gamma/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/d1_BN/gamma/m*
_output_shapes	
:*
dtype0

cond_1/Adam/d1_BN/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namecond_1/Adam/d1_BN/beta/m

,cond_1/Adam/d1_BN/beta/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/d1_BN/beta/m*
_output_shapes	
:*
dtype0

cond_1/Adam/d2_dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*.
shared_namecond_1/Adam/d2_dense/kernel/m

1cond_1/Adam/d2_dense/kernel/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/d2_dense/kernel/m* 
_output_shapes
:
*
dtype0

cond_1/Adam/d2_dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/d2_dense/bias/m

/cond_1/Adam/d2_dense/bias/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/d2_dense/bias/m*
_output_shapes	
:*
dtype0

cond_1/Adam/d2_BN/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecond_1/Adam/d2_BN/gamma/m

-cond_1/Adam/d2_BN/gamma/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/d2_BN/gamma/m*
_output_shapes	
:*
dtype0

cond_1/Adam/d2_BN/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namecond_1/Adam/d2_BN/beta/m

,cond_1/Adam/d2_BN/beta/m/Read/ReadVariableOpReadVariableOpcond_1/Adam/d2_BN/beta/m*
_output_shapes	
:*
dtype0
¡
"cond_1/Adam/dense_predict/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"cond_1/Adam/dense_predict/kernel/m

6cond_1/Adam/dense_predict/kernel/m/Read/ReadVariableOpReadVariableOp"cond_1/Adam/dense_predict/kernel/m*
_output_shapes
:	*
dtype0

 cond_1/Adam/dense_predict/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" cond_1/Adam/dense_predict/bias/m

4cond_1/Adam/dense_predict/bias/m/Read/ReadVariableOpReadVariableOp cond_1/Adam/dense_predict/bias/m*
_output_shapes
:*
dtype0
ª
%cond_1/Adam/extract_features/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%cond_1/Adam/extract_features/kernel/v
£
9cond_1/Adam/extract_features/kernel/v/Read/ReadVariableOpReadVariableOp%cond_1/Adam/extract_features/kernel/v*"
_output_shapes
:@*
dtype0

#cond_1/Adam/extract_features/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#cond_1/Adam/extract_features/bias/v

7cond_1/Adam/extract_features/bias/v/Read/ReadVariableOpReadVariableOp#cond_1/Adam/extract_features/bias/v*
_output_shapes
:@*
dtype0
¦
'cond_1/Adam/extract_features_BN/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'cond_1/Adam/extract_features_BN/gamma/v

;cond_1/Adam/extract_features_BN/gamma/v/Read/ReadVariableOpReadVariableOp'cond_1/Adam/extract_features_BN/gamma/v*
_output_shapes
:@*
dtype0
¤
&cond_1/Adam/extract_features_BN/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&cond_1/Adam/extract_features_BN/beta/v

:cond_1/Adam/extract_features_BN/beta/v/Read/ReadVariableOpReadVariableOp&cond_1/Adam/extract_features_BN/beta/v*
_output_shapes
:@*
dtype0

cond_1/Adam/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*+
shared_namecond_1/Adam/conv1/kernel/v

.cond_1/Adam/conv1/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1/kernel/v*"
_output_shapes
:@@*
dtype0

cond_1/Adam/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namecond_1/Adam/conv1/bias/v

,cond_1/Adam/conv1/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1/bias/v*
_output_shapes
:@*
dtype0

cond_1/Adam/conv1_BN/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecond_1/Adam/conv1_BN/gamma/v

0cond_1/Adam/conv1_BN/gamma/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1_BN/gamma/v*
_output_shapes
:@*
dtype0

cond_1/Adam/conv1_BN/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namecond_1/Adam/conv1_BN/beta/v

/cond_1/Adam/conv1_BN/beta/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv1_BN/beta/v*
_output_shapes
:@*
dtype0

cond_1/Adam/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*+
shared_namecond_1/Adam/conv2/kernel/v

.cond_1/Adam/conv2/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2/kernel/v*"
_output_shapes
:@@*
dtype0

cond_1/Adam/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namecond_1/Adam/conv2/bias/v

,cond_1/Adam/conv2/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2/bias/v*
_output_shapes
:@*
dtype0

cond_1/Adam/conv2_BN/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namecond_1/Adam/conv2_BN/gamma/v

0cond_1/Adam/conv2_BN/gamma/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2_BN/gamma/v*
_output_shapes
:@*
dtype0

cond_1/Adam/conv2_BN/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namecond_1/Adam/conv2_BN/beta/v

/cond_1/Adam/conv2_BN/beta/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/conv2_BN/beta/v*
_output_shapes
:@*
dtype0

cond_1/Adam/d1_dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*.
shared_namecond_1/Adam/d1_dense/kernel/v

1cond_1/Adam/d1_dense/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/d1_dense/kernel/v*
_output_shapes
:	@*
dtype0

cond_1/Adam/d1_dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/d1_dense/bias/v

/cond_1/Adam/d1_dense/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/d1_dense/bias/v*
_output_shapes	
:*
dtype0

cond_1/Adam/d1_BN/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecond_1/Adam/d1_BN/gamma/v

-cond_1/Adam/d1_BN/gamma/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/d1_BN/gamma/v*
_output_shapes	
:*
dtype0

cond_1/Adam/d1_BN/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namecond_1/Adam/d1_BN/beta/v

,cond_1/Adam/d1_BN/beta/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/d1_BN/beta/v*
_output_shapes	
:*
dtype0

cond_1/Adam/d2_dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*.
shared_namecond_1/Adam/d2_dense/kernel/v

1cond_1/Adam/d2_dense/kernel/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/d2_dense/kernel/v* 
_output_shapes
:
*
dtype0

cond_1/Adam/d2_dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namecond_1/Adam/d2_dense/bias/v

/cond_1/Adam/d2_dense/bias/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/d2_dense/bias/v*
_output_shapes	
:*
dtype0

cond_1/Adam/d2_BN/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namecond_1/Adam/d2_BN/gamma/v

-cond_1/Adam/d2_BN/gamma/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/d2_BN/gamma/v*
_output_shapes	
:*
dtype0

cond_1/Adam/d2_BN/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namecond_1/Adam/d2_BN/beta/v

,cond_1/Adam/d2_BN/beta/v/Read/ReadVariableOpReadVariableOpcond_1/Adam/d2_BN/beta/v*
_output_shapes	
:*
dtype0
¡
"cond_1/Adam/dense_predict/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"cond_1/Adam/dense_predict/kernel/v

6cond_1/Adam/dense_predict/kernel/v/Read/ReadVariableOpReadVariableOp"cond_1/Adam/dense_predict/kernel/v*
_output_shapes
:	*
dtype0

 cond_1/Adam/dense_predict/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" cond_1/Adam/dense_predict/bias/v

4cond_1/Adam/dense_predict/bias/v/Read/ReadVariableOpReadVariableOp cond_1/Adam/dense_predict/bias/v*
_output_shapes
:*
dtype0

ConstConst*
_output_shapes
:*
dtype0*I
value@B>"4=H@ñÊ@üý1B¶ ?fØALW?Ì@1qhB¤@¨/f@:¹?  ?ág@

NoOpNoOp
Ð
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*
valueýBù Bñ
Ö
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer-18
layer-19
layer_with_weights-8
layer-20
layer_with_weights-9
layer-21
layer-22
layer_with_weights-10
layer-23
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api

%axis
	&gamma
'beta
(moving_mean
)moving_variance
*	variables
+trainable_variables
,regularization_losses
-	keras_api
R
.	variables
/trainable_variables
0regularization_losses
1	keras_api
R
2	variables
3trainable_variables
4regularization_losses
5	keras_api
h

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api

<axis
	=gamma
>beta
?moving_mean
@moving_variance
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
R
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
R
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
R
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
h

Qkernel
Rbias
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api

Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
\	variables
]trainable_variables
^regularization_losses
_	keras_api
R
`	variables
atrainable_variables
bregularization_losses
c	keras_api
R
d	variables
etrainable_variables
fregularization_losses
g	keras_api
R
h	variables
itrainable_variables
jregularization_losses
k	keras_api
R
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
h

pkernel
qbias
r	variables
strainable_variables
tregularization_losses
u	keras_api

vaxis
	wgamma
xbeta
ymoving_mean
zmoving_variance
{	variables
|trainable_variables
}regularization_losses
~	keras_api
U
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
n
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
¯
 
loss_scale
¡base_optimizer
¢beta_1
£beta_2

¤decay
¥learning_rate
	¦iterm­ m®&m¯'m°6m±7m²=m³>m´QmµRm¶Xm·Ym¸pm¹qmºwm»xm¼	m½	m¾	m¿	mÀ	mÁ	mÂvÃ vÄ&vÅ'vÆ6vÇ7vÈ=vÉ>vÊQvËRvÌXvÍYvÎpvÏqvÐwvÑxvÒ	vÓ	vÔ	vÕ	vÖ	v×	vØ
þ
0
 1
&2
'3
(4
)5
66
77
=8
>9
?10
@11
Q12
R13
X14
Y15
Z16
[17
p18
q19
w20
x21
y22
z23
24
25
26
27
28
29
30
31
¬
0
 1
&2
'3
64
75
=6
>7
Q8
R9
X10
Y11
p12
q13
w14
x15
16
17
18
19
20
21
 
²
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
	variables
trainable_variables
regularization_losses
 
ca
VARIABLE_VALUEextract_features/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEextract_features/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1

0
 1
 
²
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
!	variables
"trainable_variables
#regularization_losses
 
db
VARIABLE_VALUEextract_features_BN/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEextract_features_BN/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEextract_features_BN/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#extract_features_BN/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
(2
)3

&0
'1
 
²
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
*	variables
+trainable_variables
,regularization_losses
 
 
 
²
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
.	variables
/trainable_variables
0regularization_losses
 
 
 
²
»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
2	variables
3trainable_variables
4regularization_losses
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
²
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
8	variables
9trainable_variables
:regularization_losses
 
YW
VARIABLE_VALUEconv1_BN/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1_BN/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEconv1_BN/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEconv1_BN/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

=0
>1
?2
@3

=0
>1
 
²
Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
 
 
 
²
Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
 
 
 
²
Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
 
 
 
²
Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1

Q0
R1
 
²
Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
 
YW
VARIABLE_VALUEconv2_BN/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2_BN/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEconv2_BN/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEconv2_BN/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

X0
Y1
Z2
[3

X0
Y1
 
²
Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
\	variables
]trainable_variables
^regularization_losses
 
 
 
²
ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
`	variables
atrainable_variables
bregularization_losses
 
 
 
²
ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
d	variables
etrainable_variables
fregularization_losses
 
 
 
²
ínon_trainable_variables
îlayers
ïmetrics
 ðlayer_regularization_losses
ñlayer_metrics
h	variables
itrainable_variables
jregularization_losses
 
 
 
²
ònon_trainable_variables
ólayers
ômetrics
 õlayer_regularization_losses
ölayer_metrics
l	variables
mtrainable_variables
nregularization_losses
[Y
VARIABLE_VALUEd1_dense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEd1_dense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

p0
q1

p0
q1
 
²
÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
r	variables
strainable_variables
tregularization_losses
 
VT
VARIABLE_VALUEd1_BN/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
d1_BN/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEd1_BN/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEd1_BN/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

w0
x1
y2
z3

w0
x1
 
²
ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
{	variables
|trainable_variables
}regularization_losses
 
 
 
´
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
[Y
VARIABLE_VALUEd2_dense/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEd2_dense/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 
VT
VARIABLE_VALUEd2_BN/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
d2_BN/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEd2_BN/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEd2_BN/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
0
1
2
3

0
1
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
a_
VARIABLE_VALUEdense_predict/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEdense_predict/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 
 
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcond_1/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
H
(0
)1
?2
@3
Z4
[5
y6
z7
8
9
¶
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23

¡0
¢1
 
 
 
 
 
 
 

(0
)1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
@1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Z0
[1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

y0
z1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
jh
VARIABLE_VALUEcurrent_loss_scaleBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUE
good_steps:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUE
8

£total

¤count
¥	variables
¦	keras_api
S

§drugs
	¨AUCs
©Ns
ª_call_result
«	variables
¬	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

£0
¤1

¥	variables
 
MK
VARIABLE_VALUEAUCs3keras_api/metrics/1/AUCs/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUENs1keras_api/metrics/1/Ns/.ATTRIBUTES/VARIABLE_VALUE
 

¨0
©1

«	variables

VARIABLE_VALUE%cond_1/Adam/extract_features/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#cond_1/Adam/extract_features/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'cond_1/Adam/extract_features_BN/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&cond_1/Adam/extract_features_BN/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/conv1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEcond_1/Adam/conv1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/conv1_BN/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/conv1_BN/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/conv2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEcond_1/Adam/conv2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/conv2_BN/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/conv2_BN/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/d1_dense/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/d1_dense/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEcond_1/Adam/d1_BN/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEcond_1/Adam/d1_BN/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/d2_dense/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/d2_dense/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEcond_1/Adam/d2_BN/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEcond_1/Adam/d2_BN/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"cond_1/Adam/dense_predict/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE cond_1/Adam/dense_predict/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%cond_1/Adam/extract_features/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#cond_1/Adam/extract_features/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE'cond_1/Adam/extract_features_BN/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&cond_1/Adam/extract_features_BN/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/conv1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEcond_1/Adam/conv1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/conv1_BN/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/conv1_BN/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/conv2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEcond_1/Adam/conv2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/conv2_BN/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/conv2_BN/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/d1_dense/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/d1_dense/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEcond_1/Adam/d1_BN/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEcond_1/Adam/d1_BN/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/d2_dense/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEcond_1/Adam/d2_dense/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEcond_1/Adam/d2_BN/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEcond_1/Adam/d2_BN/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"cond_1/Adam/dense_predict/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE cond_1/Adam/dense_predict/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_inputPlaceholder*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*)
shape :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
à
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputextract_features/kernelextract_features/bias#extract_features_BN/moving_varianceextract_features_BN/gammaextract_features_BN/moving_meanextract_features_BN/betaconv1/kernel
conv1/biasconv1_BN/moving_varianceconv1_BN/gammaconv1_BN/moving_meanconv1_BN/betaconv2/kernel
conv2/biasconv2_BN/moving_varianceconv2_BN/gammaconv2_BN/moving_meanconv2_BN/betad1_dense/kerneld1_dense/biasd1_BN/moving_varianced1_BN/gammad1_BN/moving_mean
d1_BN/betad2_dense/kerneld2_dense/biasd2_BN/moving_varianced2_BN/gammad2_BN/moving_mean
d2_BN/betadense_predict/kerneldense_predict/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_558358
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
î 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+extract_features/kernel/Read/ReadVariableOp)extract_features/bias/Read/ReadVariableOp-extract_features_BN/gamma/Read/ReadVariableOp,extract_features_BN/beta/Read/ReadVariableOp3extract_features_BN/moving_mean/Read/ReadVariableOp7extract_features_BN/moving_variance/Read/ReadVariableOp conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp"conv1_BN/gamma/Read/ReadVariableOp!conv1_BN/beta/Read/ReadVariableOp(conv1_BN/moving_mean/Read/ReadVariableOp,conv1_BN/moving_variance/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp"conv2_BN/gamma/Read/ReadVariableOp!conv2_BN/beta/Read/ReadVariableOp(conv2_BN/moving_mean/Read/ReadVariableOp,conv2_BN/moving_variance/Read/ReadVariableOp#d1_dense/kernel/Read/ReadVariableOp!d1_dense/bias/Read/ReadVariableOpd1_BN/gamma/Read/ReadVariableOpd1_BN/beta/Read/ReadVariableOp%d1_BN/moving_mean/Read/ReadVariableOp)d1_BN/moving_variance/Read/ReadVariableOp#d2_dense/kernel/Read/ReadVariableOp!d2_dense/bias/Read/ReadVariableOpd2_BN/gamma/Read/ReadVariableOpd2_BN/beta/Read/ReadVariableOp%d2_BN/moving_mean/Read/ReadVariableOp)d2_BN/moving_variance/Read/ReadVariableOp(dense_predict/kernel/Read/ReadVariableOp&dense_predict/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp$cond_1/Adam/iter/Read/ReadVariableOp&current_loss_scale/Read/ReadVariableOpgood_steps/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpAUCs/Read/ReadVariableOpNs/Read/ReadVariableOp9cond_1/Adam/extract_features/kernel/m/Read/ReadVariableOp7cond_1/Adam/extract_features/bias/m/Read/ReadVariableOp;cond_1/Adam/extract_features_BN/gamma/m/Read/ReadVariableOp:cond_1/Adam/extract_features_BN/beta/m/Read/ReadVariableOp.cond_1/Adam/conv1/kernel/m/Read/ReadVariableOp,cond_1/Adam/conv1/bias/m/Read/ReadVariableOp0cond_1/Adam/conv1_BN/gamma/m/Read/ReadVariableOp/cond_1/Adam/conv1_BN/beta/m/Read/ReadVariableOp.cond_1/Adam/conv2/kernel/m/Read/ReadVariableOp,cond_1/Adam/conv2/bias/m/Read/ReadVariableOp0cond_1/Adam/conv2_BN/gamma/m/Read/ReadVariableOp/cond_1/Adam/conv2_BN/beta/m/Read/ReadVariableOp1cond_1/Adam/d1_dense/kernel/m/Read/ReadVariableOp/cond_1/Adam/d1_dense/bias/m/Read/ReadVariableOp-cond_1/Adam/d1_BN/gamma/m/Read/ReadVariableOp,cond_1/Adam/d1_BN/beta/m/Read/ReadVariableOp1cond_1/Adam/d2_dense/kernel/m/Read/ReadVariableOp/cond_1/Adam/d2_dense/bias/m/Read/ReadVariableOp-cond_1/Adam/d2_BN/gamma/m/Read/ReadVariableOp,cond_1/Adam/d2_BN/beta/m/Read/ReadVariableOp6cond_1/Adam/dense_predict/kernel/m/Read/ReadVariableOp4cond_1/Adam/dense_predict/bias/m/Read/ReadVariableOp9cond_1/Adam/extract_features/kernel/v/Read/ReadVariableOp7cond_1/Adam/extract_features/bias/v/Read/ReadVariableOp;cond_1/Adam/extract_features_BN/gamma/v/Read/ReadVariableOp:cond_1/Adam/extract_features_BN/beta/v/Read/ReadVariableOp.cond_1/Adam/conv1/kernel/v/Read/ReadVariableOp,cond_1/Adam/conv1/bias/v/Read/ReadVariableOp0cond_1/Adam/conv1_BN/gamma/v/Read/ReadVariableOp/cond_1/Adam/conv1_BN/beta/v/Read/ReadVariableOp.cond_1/Adam/conv2/kernel/v/Read/ReadVariableOp,cond_1/Adam/conv2/bias/v/Read/ReadVariableOp0cond_1/Adam/conv2_BN/gamma/v/Read/ReadVariableOp/cond_1/Adam/conv2_BN/beta/v/Read/ReadVariableOp1cond_1/Adam/d1_dense/kernel/v/Read/ReadVariableOp/cond_1/Adam/d1_dense/bias/v/Read/ReadVariableOp-cond_1/Adam/d1_BN/gamma/v/Read/ReadVariableOp,cond_1/Adam/d1_BN/beta/v/Read/ReadVariableOp1cond_1/Adam/d2_dense/kernel/v/Read/ReadVariableOp/cond_1/Adam/d2_dense/bias/v/Read/ReadVariableOp-cond_1/Adam/d2_BN/gamma/v/Read/ReadVariableOp,cond_1/Adam/d2_BN/beta/v/Read/ReadVariableOp6cond_1/Adam/dense_predict/kernel/v/Read/ReadVariableOp4cond_1/Adam/dense_predict/bias/v/Read/ReadVariableOpConst_1*d
Tin]
[2Y		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_559990

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameextract_features/kernelextract_features/biasextract_features_BN/gammaextract_features_BN/betaextract_features_BN/moving_mean#extract_features_BN/moving_varianceconv1/kernel
conv1/biasconv1_BN/gammaconv1_BN/betaconv1_BN/moving_meanconv1_BN/moving_varianceconv2/kernel
conv2/biasconv2_BN/gammaconv2_BN/betaconv2_BN/moving_meanconv2_BN/moving_varianced1_dense/kerneld1_dense/biasd1_BN/gamma
d1_BN/betad1_BN/moving_meand1_BN/moving_varianced2_dense/kerneld2_dense/biasd2_BN/gamma
d2_BN/betad2_BN/moving_meand2_BN/moving_variancedense_predict/kerneldense_predict/biasbeta_1beta_2decaylearning_ratecond_1/Adam/itercurrent_loss_scale
good_stepstotalcountAUCsNs%cond_1/Adam/extract_features/kernel/m#cond_1/Adam/extract_features/bias/m'cond_1/Adam/extract_features_BN/gamma/m&cond_1/Adam/extract_features_BN/beta/mcond_1/Adam/conv1/kernel/mcond_1/Adam/conv1/bias/mcond_1/Adam/conv1_BN/gamma/mcond_1/Adam/conv1_BN/beta/mcond_1/Adam/conv2/kernel/mcond_1/Adam/conv2/bias/mcond_1/Adam/conv2_BN/gamma/mcond_1/Adam/conv2_BN/beta/mcond_1/Adam/d1_dense/kernel/mcond_1/Adam/d1_dense/bias/mcond_1/Adam/d1_BN/gamma/mcond_1/Adam/d1_BN/beta/mcond_1/Adam/d2_dense/kernel/mcond_1/Adam/d2_dense/bias/mcond_1/Adam/d2_BN/gamma/mcond_1/Adam/d2_BN/beta/m"cond_1/Adam/dense_predict/kernel/m cond_1/Adam/dense_predict/bias/m%cond_1/Adam/extract_features/kernel/v#cond_1/Adam/extract_features/bias/v'cond_1/Adam/extract_features_BN/gamma/v&cond_1/Adam/extract_features_BN/beta/vcond_1/Adam/conv1/kernel/vcond_1/Adam/conv1/bias/vcond_1/Adam/conv1_BN/gamma/vcond_1/Adam/conv1_BN/beta/vcond_1/Adam/conv2/kernel/vcond_1/Adam/conv2/bias/vcond_1/Adam/conv2_BN/gamma/vcond_1/Adam/conv2_BN/beta/vcond_1/Adam/d1_dense/kernel/vcond_1/Adam/d1_dense/bias/vcond_1/Adam/d1_BN/gamma/vcond_1/Adam/d1_BN/beta/vcond_1/Adam/d2_dense/kernel/vcond_1/Adam/d2_dense/bias/vcond_1/Adam/d2_BN/gamma/vcond_1/Adam/d2_BN/beta/v"cond_1/Adam/dense_predict/kernel/v cond_1/Adam/dense_predict/bias/v*c
Tin\
Z2X*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_560261ÿÝ


A__inference_conv2_layer_call_and_return_conditional_losses_557409

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
Conv1D/ExpandDims_1/CastCast*Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@@Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Conv1D/ExpandDims_1
ExpandDimsConv1D/ExpandDims_1/Cast:y:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¶
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@|
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


e
F__inference_d2_dropout_layer_call_and_return_conditional_losses_557660

inputs
identityP
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B jze
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed{Y
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B jæd§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
äb
 
A__inference_model_layer_call_and_return_conditional_losses_558188	
input-
extract_features_558099:@%
extract_features_558101:@(
extract_features_bn_558104:@(
extract_features_bn_558106:@(
extract_features_bn_558108:@(
extract_features_bn_558110:@"
conv1_558115:@@
conv1_558117:@
conv1_bn_558120:@
conv1_bn_558122:@
conv1_bn_558124:@
conv1_bn_558126:@"
conv2_558132:@@
conv2_558134:@
conv2_bn_558137:@
conv2_bn_558139:@
conv2_bn_558141:@
conv2_bn_558143:@"
d1_dense_558150:	@
d1_dense_558152:	
d1_bn_558155:	
d1_bn_558157:	
d1_bn_558159:	
d1_bn_558161:	#
d2_dense_558166:

d2_dense_558168:	
d2_bn_558171:	
d2_bn_558173:	
d2_bn_558175:	
d2_bn_558177:	'
dense_predict_558182:	"
dense_predict_558184:
identity¢conv1/StatefulPartitionedCall¢ conv1_BN/StatefulPartitionedCall¢conv2/StatefulPartitionedCall¢ conv2_BN/StatefulPartitionedCall¢d1_BN/StatefulPartitionedCall¢ d1_dense/StatefulPartitionedCall¢d2_BN/StatefulPartitionedCall¢ d2_dense/StatefulPartitionedCall¢%dense_predict/StatefulPartitionedCall¢(extract_features/StatefulPartitionedCall¢+extract_features_BN/StatefulPartitionedCallr
extract_features/CastCastinput*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ³
(extract_features/StatefulPartitionedCallStatefulPartitionedCallextract_features/Cast:y:0extract_features_558099extract_features_558101*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_extract_features_layer_call_and_return_conditional_losses_557308
+extract_features_BN/StatefulPartitionedCallStatefulPartitionedCall1extract_features/StatefulPartitionedCall:output:0extract_features_bn_558104extract_features_bn_558106extract_features_bn_558108extract_features_bn_558110*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_extract_features_BN_layer_call_and_return_conditional_losses_556836
%extract_features_RELU/PartitionedCallPartitionedCall4extract_features_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_extract_features_RELU_layer_call_and_return_conditional_losses_557328ú
conv1_dropout/PartitionedCallPartitionedCall.extract_features_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv1_dropout_layer_call_and_return_conditional_losses_557335
conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1_dropout/PartitionedCall:output:0conv1_558115conv1_558117*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_557354Æ
 conv1_BN/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv1_bn_558120conv1_bn_558122conv1_bn_558124conv1_bn_558126*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1_BN_layer_call_and_return_conditional_losses_556922ï
conv1_RELU/PartitionedCallPartitionedCall)conv1_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv1_RELU_layer_call_and_return_conditional_losses_557374å
conv1_mp/PartitionedCallPartitionedCall#conv1_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1_mp_layer_call_and_return_conditional_losses_557383í
conv2_dropout/PartitionedCallPartitionedCall!conv1_mp/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2_dropout_layer_call_and_return_conditional_losses_557390
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv2_dropout/PartitionedCall:output:0conv2_558132conv2_558134*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_557409Æ
 conv2_BN/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0conv2_bn_558137conv2_bn_558139conv2_bn_558141conv2_bn_558143*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2_BN_layer_call_and_return_conditional_losses_557023ï
conv2_RELU/PartitionedCallPartitionedCall)conv2_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2_RELU_layer_call_and_return_conditional_losses_557429å
conv2_mp/PartitionedCallPartitionedCall#conv2_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2_mp_layer_call_and_return_conditional_losses_557438æ
 combine_features/PartitionedCallPartitionedCall!conv2_mp/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_combine_features_layer_call_and_return_conditional_losses_557445â
d1_dropout/PartitionedCallPartitionedCall)combine_features/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_d1_dropout_layer_call_and_return_conditional_losses_557452
 d1_dense/StatefulPartitionedCallStatefulPartitionedCall#d1_dropout/PartitionedCall:output:0d1_dense_558150d1_dense_558152*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_d1_dense_layer_call_and_return_conditional_losses_557466«
d1_BN/StatefulPartitionedCallStatefulPartitionedCall)d1_dense/StatefulPartitionedCall:output:0d1_bn_558155d1_bn_558157d1_bn_558159d1_bn_558161*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_d1_BN_layer_call_and_return_conditional_losses_557137Ú
d1_RELU/PartitionedCallPartitionedCall&d1_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_d1_RELU_layer_call_and_return_conditional_losses_557486Ú
d2_dropout/PartitionedCallPartitionedCall d1_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_d2_dropout_layer_call_and_return_conditional_losses_557493
 d2_dense/StatefulPartitionedCallStatefulPartitionedCall#d2_dropout/PartitionedCall:output:0d2_dense_558166d2_dense_558168*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_d2_dense_layer_call_and_return_conditional_losses_557507«
d2_BN/StatefulPartitionedCallStatefulPartitionedCall)d2_dense/StatefulPartitionedCall:output:0d2_bn_558171d2_bn_558173d2_bn_558175d2_bn_558177*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_d2_BN_layer_call_and_return_conditional_losses_557223Ú
d2_RELU/PartitionedCallPartitionedCall&d2_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_d2_RELU_layer_call_and_return_conditional_losses_557527~
dense_predict/CastCast d2_RELU/PartitionedCall:output:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%dense_predict/StatefulPartitionedCallStatefulPartitionedCalldense_predict/Cast:y:0dense_predict_558182dense_predict_558184*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_predict_layer_call_and_return_conditional_losses_557540}
IdentityIdentity.dense_predict/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
NoOpNoOp^conv1/StatefulPartitionedCall!^conv1_BN/StatefulPartitionedCall^conv2/StatefulPartitionedCall!^conv2_BN/StatefulPartitionedCall^d1_BN/StatefulPartitionedCall!^d1_dense/StatefulPartitionedCall^d2_BN/StatefulPartitionedCall!^d2_dense/StatefulPartitionedCall&^dense_predict/StatefulPartitionedCall)^extract_features/StatefulPartitionedCall,^extract_features_BN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2D
 conv1_BN/StatefulPartitionedCall conv1_BN/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2D
 conv2_BN/StatefulPartitionedCall conv2_BN/StatefulPartitionedCall2>
d1_BN/StatefulPartitionedCalld1_BN/StatefulPartitionedCall2D
 d1_dense/StatefulPartitionedCall d1_dense/StatefulPartitionedCall2>
d2_BN/StatefulPartitionedCalld2_BN/StatefulPartitionedCall2D
 d2_dense/StatefulPartitionedCall d2_dense/StatefulPartitionedCall2N
%dense_predict/StatefulPartitionedCall%dense_predict/StatefulPartitionedCall2T
(extract_features/StatefulPartitionedCall(extract_features/StatefulPartitionedCall2Z
+extract_features_BN/StatefulPartitionedCall+extract_features_BN/StatefulPartitionedCall:[ W
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
Û
h
L__inference_combine_features_layer_call_and_return_conditional_losses_559402

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :d
MaxMaxinputsMax/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
IdentityIdentityMax:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ë
_
C__inference_d2_RELU_layer_call_and_return_conditional_losses_559686

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
h
L__inference_combine_features_layer_call_and_return_conditional_losses_559396

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
I__inference_conv2_dropout_layer_call_and_return_conditional_losses_557390

inputs

identity_1[
IdentityIdentityinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
çb
¡
A__inference_model_layer_call_and_return_conditional_losses_557547

inputs-
extract_features_557309:@%
extract_features_557311:@(
extract_features_bn_557314:@(
extract_features_bn_557316:@(
extract_features_bn_557318:@(
extract_features_bn_557320:@"
conv1_557355:@@
conv1_557357:@
conv1_bn_557360:@
conv1_bn_557362:@
conv1_bn_557364:@
conv1_bn_557366:@"
conv2_557410:@@
conv2_557412:@
conv2_bn_557415:@
conv2_bn_557417:@
conv2_bn_557419:@
conv2_bn_557421:@"
d1_dense_557467:	@
d1_dense_557469:	
d1_bn_557472:	
d1_bn_557474:	
d1_bn_557476:	
d1_bn_557478:	#
d2_dense_557508:

d2_dense_557510:	
d2_bn_557513:	
d2_bn_557515:	
d2_bn_557517:	
d2_bn_557519:	'
dense_predict_557541:	"
dense_predict_557543:
identity¢conv1/StatefulPartitionedCall¢ conv1_BN/StatefulPartitionedCall¢conv2/StatefulPartitionedCall¢ conv2_BN/StatefulPartitionedCall¢d1_BN/StatefulPartitionedCall¢ d1_dense/StatefulPartitionedCall¢d2_BN/StatefulPartitionedCall¢ d2_dense/StatefulPartitionedCall¢%dense_predict/StatefulPartitionedCall¢(extract_features/StatefulPartitionedCall¢+extract_features_BN/StatefulPartitionedCalls
extract_features/CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ³
(extract_features/StatefulPartitionedCallStatefulPartitionedCallextract_features/Cast:y:0extract_features_557309extract_features_557311*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_extract_features_layer_call_and_return_conditional_losses_557308
+extract_features_BN/StatefulPartitionedCallStatefulPartitionedCall1extract_features/StatefulPartitionedCall:output:0extract_features_bn_557314extract_features_bn_557316extract_features_bn_557318extract_features_bn_557320*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_extract_features_BN_layer_call_and_return_conditional_losses_556836
%extract_features_RELU/PartitionedCallPartitionedCall4extract_features_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_extract_features_RELU_layer_call_and_return_conditional_losses_557328ú
conv1_dropout/PartitionedCallPartitionedCall.extract_features_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv1_dropout_layer_call_and_return_conditional_losses_557335
conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1_dropout/PartitionedCall:output:0conv1_557355conv1_557357*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_557354Æ
 conv1_BN/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv1_bn_557360conv1_bn_557362conv1_bn_557364conv1_bn_557366*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1_BN_layer_call_and_return_conditional_losses_556922ï
conv1_RELU/PartitionedCallPartitionedCall)conv1_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv1_RELU_layer_call_and_return_conditional_losses_557374å
conv1_mp/PartitionedCallPartitionedCall#conv1_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1_mp_layer_call_and_return_conditional_losses_557383í
conv2_dropout/PartitionedCallPartitionedCall!conv1_mp/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2_dropout_layer_call_and_return_conditional_losses_557390
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv2_dropout/PartitionedCall:output:0conv2_557410conv2_557412*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_557409Æ
 conv2_BN/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0conv2_bn_557415conv2_bn_557417conv2_bn_557419conv2_bn_557421*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2_BN_layer_call_and_return_conditional_losses_557023ï
conv2_RELU/PartitionedCallPartitionedCall)conv2_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2_RELU_layer_call_and_return_conditional_losses_557429å
conv2_mp/PartitionedCallPartitionedCall#conv2_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2_mp_layer_call_and_return_conditional_losses_557438æ
 combine_features/PartitionedCallPartitionedCall!conv2_mp/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_combine_features_layer_call_and_return_conditional_losses_557445â
d1_dropout/PartitionedCallPartitionedCall)combine_features/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_d1_dropout_layer_call_and_return_conditional_losses_557452
 d1_dense/StatefulPartitionedCallStatefulPartitionedCall#d1_dropout/PartitionedCall:output:0d1_dense_557467d1_dense_557469*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_d1_dense_layer_call_and_return_conditional_losses_557466«
d1_BN/StatefulPartitionedCallStatefulPartitionedCall)d1_dense/StatefulPartitionedCall:output:0d1_bn_557472d1_bn_557474d1_bn_557476d1_bn_557478*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_d1_BN_layer_call_and_return_conditional_losses_557137Ú
d1_RELU/PartitionedCallPartitionedCall&d1_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_d1_RELU_layer_call_and_return_conditional_losses_557486Ú
d2_dropout/PartitionedCallPartitionedCall d1_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_d2_dropout_layer_call_and_return_conditional_losses_557493
 d2_dense/StatefulPartitionedCallStatefulPartitionedCall#d2_dropout/PartitionedCall:output:0d2_dense_557508d2_dense_557510*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_d2_dense_layer_call_and_return_conditional_losses_557507«
d2_BN/StatefulPartitionedCallStatefulPartitionedCall)d2_dense/StatefulPartitionedCall:output:0d2_bn_557513d2_bn_557515d2_bn_557517d2_bn_557519*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_d2_BN_layer_call_and_return_conditional_losses_557223Ú
d2_RELU/PartitionedCallPartitionedCall&d2_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_d2_RELU_layer_call_and_return_conditional_losses_557527~
dense_predict/CastCast d2_RELU/PartitionedCall:output:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%dense_predict/StatefulPartitionedCallStatefulPartitionedCalldense_predict/Cast:y:0dense_predict_557541dense_predict_557543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_predict_layer_call_and_return_conditional_losses_557540}
IdentityIdentity.dense_predict/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
NoOpNoOp^conv1/StatefulPartitionedCall!^conv1_BN/StatefulPartitionedCall^conv2/StatefulPartitionedCall!^conv2_BN/StatefulPartitionedCall^d1_BN/StatefulPartitionedCall!^d1_dense/StatefulPartitionedCall^d2_BN/StatefulPartitionedCall!^d2_dense/StatefulPartitionedCall&^dense_predict/StatefulPartitionedCall)^extract_features/StatefulPartitionedCall,^extract_features_BN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2D
 conv1_BN/StatefulPartitionedCall conv1_BN/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2D
 conv2_BN/StatefulPartitionedCall conv2_BN/StatefulPartitionedCall2>
d1_BN/StatefulPartitionedCalld1_BN/StatefulPartitionedCall2D
 d1_dense/StatefulPartitionedCall d1_dense/StatefulPartitionedCall2>
d2_BN/StatefulPartitionedCalld2_BN/StatefulPartitionedCall2D
 d2_dense/StatefulPartitionedCall d2_dense/StatefulPartitionedCall2N
%dense_predict/StatefulPartitionedCall%dense_predict/StatefulPartitionedCall2T
(extract_features/StatefulPartitionedCall(extract_features/StatefulPartitionedCall2Z
+extract_features_BN/StatefulPartitionedCall+extract_features_BN/StatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

`
D__inference_conv1_mp_layer_call_and_return_conditional_losses_559215

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :|

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¦
MaxPoolMaxPoolExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
z
SqueezeSqueezeMaxPool:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
e
IdentityIdentitySqueeze:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
J
.__inference_conv2_dropout_layer_call_fn_559220

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2_dropout_layer_call_and_return_conditional_losses_557390m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

m
Q__inference_extract_features_RELU_layer_call_and_return_conditional_losses_557328

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@g
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
Ã
&__inference_model_layer_call_fn_557614	
input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	@

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:


unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_557547o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
×e
ê
A__inference_model_layer_call_and_return_conditional_losses_558281	
input-
extract_features_558192:@%
extract_features_558194:@(
extract_features_bn_558197:@(
extract_features_bn_558199:@(
extract_features_bn_558201:@(
extract_features_bn_558203:@"
conv1_558208:@@
conv1_558210:@
conv1_bn_558213:@
conv1_bn_558215:@
conv1_bn_558217:@
conv1_bn_558219:@"
conv2_558225:@@
conv2_558227:@
conv2_bn_558230:@
conv2_bn_558232:@
conv2_bn_558234:@
conv2_bn_558236:@"
d1_dense_558243:	@
d1_dense_558245:	
d1_bn_558248:	
d1_bn_558250:	
d1_bn_558252:	
d1_bn_558254:	#
d2_dense_558259:

d2_dense_558261:	
d2_bn_558264:	
d2_bn_558266:	
d2_bn_558268:	
d2_bn_558270:	'
dense_predict_558275:	"
dense_predict_558277:
identity¢conv1/StatefulPartitionedCall¢ conv1_BN/StatefulPartitionedCall¢conv2/StatefulPartitionedCall¢ conv2_BN/StatefulPartitionedCall¢d1_BN/StatefulPartitionedCall¢ d1_dense/StatefulPartitionedCall¢"d1_dropout/StatefulPartitionedCall¢d2_BN/StatefulPartitionedCall¢ d2_dense/StatefulPartitionedCall¢"d2_dropout/StatefulPartitionedCall¢%dense_predict/StatefulPartitionedCall¢(extract_features/StatefulPartitionedCall¢+extract_features_BN/StatefulPartitionedCallr
extract_features/CastCastinput*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ³
(extract_features/StatefulPartitionedCallStatefulPartitionedCallextract_features/Cast:y:0extract_features_558192extract_features_558194*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_extract_features_layer_call_and_return_conditional_losses_557308
+extract_features_BN/StatefulPartitionedCallStatefulPartitionedCall1extract_features/StatefulPartitionedCall:output:0extract_features_bn_558197extract_features_bn_558199extract_features_bn_558201extract_features_bn_558203*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_extract_features_BN_layer_call_and_return_conditional_losses_556885
%extract_features_RELU/PartitionedCallPartitionedCall4extract_features_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_extract_features_RELU_layer_call_and_return_conditional_losses_557328ú
conv1_dropout/PartitionedCallPartitionedCall.extract_features_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv1_dropout_layer_call_and_return_conditional_losses_557776
conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1_dropout/PartitionedCall:output:0conv1_558208conv1_558210*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_557354Ä
 conv1_BN/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv1_bn_558213conv1_bn_558215conv1_bn_558217conv1_bn_558219*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1_BN_layer_call_and_return_conditional_losses_556971ï
conv1_RELU/PartitionedCallPartitionedCall)conv1_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv1_RELU_layer_call_and_return_conditional_losses_557374å
conv1_mp/PartitionedCallPartitionedCall#conv1_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1_mp_layer_call_and_return_conditional_losses_557383í
conv2_dropout/PartitionedCallPartitionedCall!conv1_mp/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2_dropout_layer_call_and_return_conditional_losses_557740
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv2_dropout/PartitionedCall:output:0conv2_558225conv2_558227*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_557409Ä
 conv2_BN/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0conv2_bn_558230conv2_bn_558232conv2_bn_558234conv2_bn_558236*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2_BN_layer_call_and_return_conditional_losses_557072ï
conv2_RELU/PartitionedCallPartitionedCall)conv2_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2_RELU_layer_call_and_return_conditional_losses_557429å
conv2_mp/PartitionedCallPartitionedCall#conv2_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2_mp_layer_call_and_return_conditional_losses_557438æ
 combine_features/PartitionedCallPartitionedCall!conv2_mp/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_combine_features_layer_call_and_return_conditional_losses_557445ò
"d1_dropout/StatefulPartitionedCallStatefulPartitionedCall)combine_features/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_d1_dropout_layer_call_and_return_conditional_losses_557699
 d1_dense/StatefulPartitionedCallStatefulPartitionedCall+d1_dropout/StatefulPartitionedCall:output:0d1_dense_558243d1_dense_558245*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_d1_dense_layer_call_and_return_conditional_losses_557466©
d1_BN/StatefulPartitionedCallStatefulPartitionedCall)d1_dense/StatefulPartitionedCall:output:0d1_bn_558248d1_bn_558250d1_bn_558252d1_bn_558254*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_d1_BN_layer_call_and_return_conditional_losses_557186Ú
d1_RELU/PartitionedCallPartitionedCall&d1_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_d1_RELU_layer_call_and_return_conditional_losses_557486
"d2_dropout/StatefulPartitionedCallStatefulPartitionedCall d1_RELU/PartitionedCall:output:0#^d1_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_d2_dropout_layer_call_and_return_conditional_losses_557660
 d2_dense/StatefulPartitionedCallStatefulPartitionedCall+d2_dropout/StatefulPartitionedCall:output:0d2_dense_558259d2_dense_558261*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_d2_dense_layer_call_and_return_conditional_losses_557507©
d2_BN/StatefulPartitionedCallStatefulPartitionedCall)d2_dense/StatefulPartitionedCall:output:0d2_bn_558264d2_bn_558266d2_bn_558268d2_bn_558270*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_d2_BN_layer_call_and_return_conditional_losses_557272Ú
d2_RELU/PartitionedCallPartitionedCall&d2_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_d2_RELU_layer_call_and_return_conditional_losses_557527~
dense_predict/CastCast d2_RELU/PartitionedCall:output:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%dense_predict/StatefulPartitionedCallStatefulPartitionedCalldense_predict/Cast:y:0dense_predict_558275dense_predict_558277*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_predict_layer_call_and_return_conditional_losses_557540}
IdentityIdentity.dense_predict/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv1/StatefulPartitionedCall!^conv1_BN/StatefulPartitionedCall^conv2/StatefulPartitionedCall!^conv2_BN/StatefulPartitionedCall^d1_BN/StatefulPartitionedCall!^d1_dense/StatefulPartitionedCall#^d1_dropout/StatefulPartitionedCall^d2_BN/StatefulPartitionedCall!^d2_dense/StatefulPartitionedCall#^d2_dropout/StatefulPartitionedCall&^dense_predict/StatefulPartitionedCall)^extract_features/StatefulPartitionedCall,^extract_features_BN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2D
 conv1_BN/StatefulPartitionedCall conv1_BN/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2D
 conv2_BN/StatefulPartitionedCall conv2_BN/StatefulPartitionedCall2>
d1_BN/StatefulPartitionedCalld1_BN/StatefulPartitionedCall2D
 d1_dense/StatefulPartitionedCall d1_dense/StatefulPartitionedCall2H
"d1_dropout/StatefulPartitionedCall"d1_dropout/StatefulPartitionedCall2>
d2_BN/StatefulPartitionedCalld2_BN/StatefulPartitionedCall2D
 d2_dense/StatefulPartitionedCall d2_dense/StatefulPartitionedCall2H
"d2_dropout/StatefulPartitionedCall"d2_dropout/StatefulPartitionedCall2N
%dense_predict/StatefulPartitionedCall%dense_predict/StatefulPartitionedCall2T
(extract_features/StatefulPartitionedCall(extract_features/StatefulPartitionedCall2Z
+extract_features_BN/StatefulPartitionedCall+extract_features_BN/StatefulPartitionedCall:[ W
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput

¢
1__inference_extract_features_layer_call_fn_558939

inputs
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_extract_features_layer_call_and_return_conditional_losses_557308|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
G
+__inference_d1_dropout_layer_call_fn_559407

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_d1_dropout_layer_call_and_return_conditional_losses_557452`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ô
E
)__inference_conv2_mp_layer_call_fn_559364

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2_mp_layer_call_and_return_conditional_losses_557438m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
þ
b
F__inference_conv2_RELU_layer_call_and_return_conditional_losses_557429

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@g
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ø
E
)__inference_conv1_mp_layer_call_fn_559194

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1_mp_layer_call_and_return_conditional_losses_556994v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
Ï
4__inference_extract_features_BN_layer_call_fn_558969

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_extract_features_BN_layer_call_and_return_conditional_losses_556836|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ï'
è
O__inference_extract_features_BN_layer_call_and_return_conditional_losses_556885

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpb
CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@o
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ¢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@r
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@q
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
IdentityIdentity
Cast_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Å
Ä
)__inference_conv1_BN_layer_call_fn_559108

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1_BN_layer_call_and_return_conditional_losses_556922|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 
e
I__inference_conv2_dropout_layer_call_and_return_conditional_losses_557740

inputs
identity[
IdentityIdentityinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ã
Ä
)__inference_conv1_BN_layer_call_fn_559121

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1_BN_layer_call_and_return_conditional_losses_556971|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
÷

&__inference_conv2_layer_call_fn_559243

inputs
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_557409|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¢
D
(__inference_d2_RELU_layer_call_fn_559681

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_d2_RELU_layer_call_and_return_conditional_losses_557527a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
J
.__inference_conv2_dropout_layer_call_fn_559225

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2_dropout_layer_call_and_return_conditional_losses_557740m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ù
d
F__inference_d1_dropout_layer_call_and_return_conditional_losses_559417

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
þ
b
F__inference_conv1_RELU_layer_call_and_return_conditional_losses_559189

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@g
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ý
d
F__inference_d2_dropout_layer_call_and_return_conditional_losses_557493

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

m
Q__inference_extract_features_RELU_layer_call_and_return_conditional_losses_559050

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@g
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Úe
ë
A__inference_model_layer_call_and_return_conditional_losses_557959

inputs-
extract_features_557870:@%
extract_features_557872:@(
extract_features_bn_557875:@(
extract_features_bn_557877:@(
extract_features_bn_557879:@(
extract_features_bn_557881:@"
conv1_557886:@@
conv1_557888:@
conv1_bn_557891:@
conv1_bn_557893:@
conv1_bn_557895:@
conv1_bn_557897:@"
conv2_557903:@@
conv2_557905:@
conv2_bn_557908:@
conv2_bn_557910:@
conv2_bn_557912:@
conv2_bn_557914:@"
d1_dense_557921:	@
d1_dense_557923:	
d1_bn_557926:	
d1_bn_557928:	
d1_bn_557930:	
d1_bn_557932:	#
d2_dense_557937:

d2_dense_557939:	
d2_bn_557942:	
d2_bn_557944:	
d2_bn_557946:	
d2_bn_557948:	'
dense_predict_557953:	"
dense_predict_557955:
identity¢conv1/StatefulPartitionedCall¢ conv1_BN/StatefulPartitionedCall¢conv2/StatefulPartitionedCall¢ conv2_BN/StatefulPartitionedCall¢d1_BN/StatefulPartitionedCall¢ d1_dense/StatefulPartitionedCall¢"d1_dropout/StatefulPartitionedCall¢d2_BN/StatefulPartitionedCall¢ d2_dense/StatefulPartitionedCall¢"d2_dropout/StatefulPartitionedCall¢%dense_predict/StatefulPartitionedCall¢(extract_features/StatefulPartitionedCall¢+extract_features_BN/StatefulPartitionedCalls
extract_features/CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ³
(extract_features/StatefulPartitionedCallStatefulPartitionedCallextract_features/Cast:y:0extract_features_557870extract_features_557872*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_extract_features_layer_call_and_return_conditional_losses_557308
+extract_features_BN/StatefulPartitionedCallStatefulPartitionedCall1extract_features/StatefulPartitionedCall:output:0extract_features_bn_557875extract_features_bn_557877extract_features_bn_557879extract_features_bn_557881*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_extract_features_BN_layer_call_and_return_conditional_losses_556885
%extract_features_RELU/PartitionedCallPartitionedCall4extract_features_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_extract_features_RELU_layer_call_and_return_conditional_losses_557328ú
conv1_dropout/PartitionedCallPartitionedCall.extract_features_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv1_dropout_layer_call_and_return_conditional_losses_557776
conv1/StatefulPartitionedCallStatefulPartitionedCall&conv1_dropout/PartitionedCall:output:0conv1_557886conv1_557888*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_557354Ä
 conv1_BN/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv1_bn_557891conv1_bn_557893conv1_bn_557895conv1_bn_557897*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1_BN_layer_call_and_return_conditional_losses_556971ï
conv1_RELU/PartitionedCallPartitionedCall)conv1_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv1_RELU_layer_call_and_return_conditional_losses_557374å
conv1_mp/PartitionedCallPartitionedCall#conv1_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1_mp_layer_call_and_return_conditional_losses_557383í
conv2_dropout/PartitionedCallPartitionedCall!conv1_mp/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2_dropout_layer_call_and_return_conditional_losses_557740
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv2_dropout/PartitionedCall:output:0conv2_557903conv2_557905*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2_layer_call_and_return_conditional_losses_557409Ä
 conv2_BN/StatefulPartitionedCallStatefulPartitionedCall&conv2/StatefulPartitionedCall:output:0conv2_bn_557908conv2_bn_557910conv2_bn_557912conv2_bn_557914*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2_BN_layer_call_and_return_conditional_losses_557072ï
conv2_RELU/PartitionedCallPartitionedCall)conv2_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2_RELU_layer_call_and_return_conditional_losses_557429å
conv2_mp/PartitionedCallPartitionedCall#conv2_RELU/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2_mp_layer_call_and_return_conditional_losses_557438æ
 combine_features/PartitionedCallPartitionedCall!conv2_mp/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_combine_features_layer_call_and_return_conditional_losses_557445ò
"d1_dropout/StatefulPartitionedCallStatefulPartitionedCall)combine_features/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_d1_dropout_layer_call_and_return_conditional_losses_557699
 d1_dense/StatefulPartitionedCallStatefulPartitionedCall+d1_dropout/StatefulPartitionedCall:output:0d1_dense_557921d1_dense_557923*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_d1_dense_layer_call_and_return_conditional_losses_557466©
d1_BN/StatefulPartitionedCallStatefulPartitionedCall)d1_dense/StatefulPartitionedCall:output:0d1_bn_557926d1_bn_557928d1_bn_557930d1_bn_557932*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_d1_BN_layer_call_and_return_conditional_losses_557186Ú
d1_RELU/PartitionedCallPartitionedCall&d1_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_d1_RELU_layer_call_and_return_conditional_losses_557486
"d2_dropout/StatefulPartitionedCallStatefulPartitionedCall d1_RELU/PartitionedCall:output:0#^d1_dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_d2_dropout_layer_call_and_return_conditional_losses_557660
 d2_dense/StatefulPartitionedCallStatefulPartitionedCall+d2_dropout/StatefulPartitionedCall:output:0d2_dense_557937d2_dense_557939*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_d2_dense_layer_call_and_return_conditional_losses_557507©
d2_BN/StatefulPartitionedCallStatefulPartitionedCall)d2_dense/StatefulPartitionedCall:output:0d2_bn_557942d2_bn_557944d2_bn_557946d2_bn_557948*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_d2_BN_layer_call_and_return_conditional_losses_557272Ú
d2_RELU/PartitionedCallPartitionedCall&d2_BN/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_d2_RELU_layer_call_and_return_conditional_losses_557527~
dense_predict/CastCast d2_RELU/PartitionedCall:output:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%dense_predict/StatefulPartitionedCallStatefulPartitionedCalldense_predict/Cast:y:0dense_predict_557953dense_predict_557955*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_predict_layer_call_and_return_conditional_losses_557540}
IdentityIdentity.dense_predict/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv1/StatefulPartitionedCall!^conv1_BN/StatefulPartitionedCall^conv2/StatefulPartitionedCall!^conv2_BN/StatefulPartitionedCall^d1_BN/StatefulPartitionedCall!^d1_dense/StatefulPartitionedCall#^d1_dropout/StatefulPartitionedCall^d2_BN/StatefulPartitionedCall!^d2_dense/StatefulPartitionedCall#^d2_dropout/StatefulPartitionedCall&^dense_predict/StatefulPartitionedCall)^extract_features/StatefulPartitionedCall,^extract_features_BN/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2D
 conv1_BN/StatefulPartitionedCall conv1_BN/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2D
 conv2_BN/StatefulPartitionedCall conv2_BN/StatefulPartitionedCall2>
d1_BN/StatefulPartitionedCalld1_BN/StatefulPartitionedCall2D
 d1_dense/StatefulPartitionedCall d1_dense/StatefulPartitionedCall2H
"d1_dropout/StatefulPartitionedCall"d1_dropout/StatefulPartitionedCall2>
d2_BN/StatefulPartitionedCalld2_BN/StatefulPartitionedCall2D
 d2_dense/StatefulPartitionedCall d2_dense/StatefulPartitionedCall2H
"d2_dropout/StatefulPartitionedCall"d2_dropout/StatefulPartitionedCall2N
%dense_predict/StatefulPartitionedCall%dense_predict/StatefulPartitionedCall2T
(extract_features/StatefulPartitionedCall(extract_features/StatefulPartitionedCall2Z
+extract_features_BN/StatefulPartitionedCall+extract_features_BN/StatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

`
D__inference_conv2_mp_layer_call_and_return_conditional_losses_557438

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :|

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¦
MaxPoolMaxPoolExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
z
SqueezeSqueezeMaxPool:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
e
IdentityIdentitySqueeze:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

Å
&__inference_d1_BN_layer_call_fn_559476

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_d1_BN_layer_call_and_return_conditional_losses_557186p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


L__inference_extract_features_layer_call_and_return_conditional_losses_557308

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0
Conv1D/ExpandDims_1/CastCast*Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Conv1D/ExpandDims_1
ExpandDimsConv1D/ExpandDims_1/Cast:y:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@¶
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@|
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
®
O__inference_extract_features_BN_layer_call_and_return_conditional_losses_556836

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpb
CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@r
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@q
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
IdentityIdentity
Cast_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


A__inference_conv2_layer_call_and_return_conditional_losses_559260

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
Conv1D/ExpandDims_1/CastCast*Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@@Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Conv1D/ExpandDims_1
ExpandDimsConv1D/ExpandDims_1/Cast:y:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¶
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@|
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¤
A__inference_d1_BN_layer_call_and_return_conditional_losses_557137

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:f
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity
Cast_1:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
£
D__inference_conv2_BN_layer_call_and_return_conditional_losses_559308

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpb
CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@r
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@q
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
IdentityIdentity
Cast_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ú
d
+__inference_d2_dropout_layer_call_fn_559554

inputs
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_d2_dropout_layer_call_and_return_conditional_losses_557660p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
Ä
&__inference_model_layer_call_fn_558496

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	@

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:


unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_557959o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
Ã
&__inference_model_layer_call_fn_558095	
input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	@

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:


unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_557959o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput

`
D__inference_conv2_mp_layer_call_and_return_conditional_losses_559380

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :|

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¦
MaxPoolMaxPoolExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
z
SqueezeSqueezeMaxPool:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
e
IdentityIdentitySqueeze:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Î
m
1__inference_weighted_masked_BCE_from_logits_12083

y_true
y_pred_logits
weights
identityH
IsNanIsNany_true*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿL

LogicalNot
LogicalNot	IsNan:y:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
WhereWhereLogicalNot:y:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
GatherNdGatherNdy_trueWhere:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

GatherNd_1GatherNdy_pred_logitsWhere:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÿ
strided_sliceStridedSliceWhere:index:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
shrink_axis_maskO
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
GatherV2GatherV2weightsstrided_slice:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
logistic_loss/zeros_like	ZerosLikeGatherNd_1:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
logistic_loss/GreaterEqualGreaterEqualGatherNd_1:output:0logistic_loss/zeros_like:y:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
logistic_loss/SelectSelectlogistic_loss/GreaterEqual:z:0GatherNd_1:output:0logistic_loss/zeros_like:y:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
logistic_loss/NegNegGatherNd_1:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
logistic_loss/Select_1Selectlogistic_loss/GreaterEqual:z:0logistic_loss/Neg:y:0GatherNd_1:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
logistic_loss/mulMulGatherNd_1:output:0GatherNd:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
logistic_loss/subSublogistic_loss/Select:output:0logistic_loss/mul:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
logistic_loss/ExpExplogistic_loss/Select_1:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
logistic_loss/Log1pLog1plogistic_loss/Exp:y:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
logistic_lossAddV2logistic_loss/sub:z:0logistic_loss/Log1p:y:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mulMulGatherV2:output:0logistic_loss:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿK
IdentityIdentitymul:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namey_true:VR
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_namey_pred_logits:C?

_output_shapes
:
!
_user_specified_name	weights
Ò
`
D__inference_conv1_mp_layer_call_and_return_conditional_losses_559207

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¯
MaxPoolMaxPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
_
C__inference_d2_RELU_layer_call_and_return_conditional_losses_557527

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
D
(__inference_d1_RELU_layer_call_fn_559539

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_d1_RELU_layer_call_and_return_conditional_losses_557486a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
b
F__inference_conv1_RELU_layer_call_and_return_conditional_losses_557374

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@g
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Û&
Þ
A__inference_d2_BN_layer_call_and_return_conditional_losses_557272

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:f
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity
Cast_1:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
M
1__inference_combine_features_layer_call_fn_559390

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_combine_features_layer_call_and_return_conditional_losses_557445`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


A__inference_conv1_layer_call_and_return_conditional_losses_559095

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
Conv1D/ExpandDims_1/CastCast*Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@@Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Conv1D/ExpandDims_1
ExpandDimsConv1D/ExpandDims_1/Cast:y:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¶
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@|
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ø
G
+__inference_conv1_RELU_layer_call_fn_559184

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv1_RELU_layer_call_and_return_conditional_losses_557374m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ø
G
+__inference_conv2_RELU_layer_call_fn_559349

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2_RELU_layer_call_and_return_conditional_losses_557429m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ÿ
h
L__inference_combine_features_layer_call_and_return_conditional_losses_557108

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ø
D__inference_d2_dense_layer_call_and_return_conditional_losses_557507

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0l
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
\
MatMulMatMulinputsMatMul/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:i
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
d
F__inference_d2_dropout_layer_call_and_return_conditional_losses_559559

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û
h
L__inference_combine_features_layer_call_and_return_conditional_losses_557445

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :d
MaxMaxinputsMax/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@T
IdentityIdentityMax:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
î
R
6__inference_extract_features_RELU_layer_call_fn_559045

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_extract_features_RELU_layer_call_and_return_conditional_losses_557328m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Û&
Þ
A__inference_d1_BN_layer_call_and_return_conditional_losses_559534

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:f
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity
Cast_1:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
_
C__inference_d1_RELU_layer_call_and_return_conditional_losses_559544

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

)__inference_d2_dense_layer_call_fn_559580

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_d2_dense_layer_call_and_return_conditional_losses_557507p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
£
D__inference_conv1_BN_layer_call_and_return_conditional_losses_556922

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpb
CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@r
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@q
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
IdentityIdentity
Cast_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¤
A__inference_d1_BN_layer_call_and_return_conditional_losses_559498

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:f
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity
Cast_1:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
E
)__inference_conv1_mp_layer_call_fn_559199

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1_mp_layer_call_and_return_conditional_losses_557383m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


e
F__inference_d2_dropout_layer_call_and_return_conditional_losses_559571

inputs
identityP
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B jze
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed{Y
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B jæd§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾§
¢&
__inference__traced_save_559990
file_prefix6
2savev2_extract_features_kernel_read_readvariableop4
0savev2_extract_features_bias_read_readvariableop8
4savev2_extract_features_bn_gamma_read_readvariableop7
3savev2_extract_features_bn_beta_read_readvariableop>
:savev2_extract_features_bn_moving_mean_read_readvariableopB
>savev2_extract_features_bn_moving_variance_read_readvariableop+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop-
)savev2_conv1_bn_gamma_read_readvariableop,
(savev2_conv1_bn_beta_read_readvariableop3
/savev2_conv1_bn_moving_mean_read_readvariableop7
3savev2_conv1_bn_moving_variance_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop-
)savev2_conv2_bn_gamma_read_readvariableop,
(savev2_conv2_bn_beta_read_readvariableop3
/savev2_conv2_bn_moving_mean_read_readvariableop7
3savev2_conv2_bn_moving_variance_read_readvariableop.
*savev2_d1_dense_kernel_read_readvariableop,
(savev2_d1_dense_bias_read_readvariableop*
&savev2_d1_bn_gamma_read_readvariableop)
%savev2_d1_bn_beta_read_readvariableop0
,savev2_d1_bn_moving_mean_read_readvariableop4
0savev2_d1_bn_moving_variance_read_readvariableop.
*savev2_d2_dense_kernel_read_readvariableop,
(savev2_d2_dense_bias_read_readvariableop*
&savev2_d2_bn_gamma_read_readvariableop)
%savev2_d2_bn_beta_read_readvariableop0
,savev2_d2_bn_moving_mean_read_readvariableop4
0savev2_d2_bn_moving_variance_read_readvariableop3
/savev2_dense_predict_kernel_read_readvariableop1
-savev2_dense_predict_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop/
+savev2_cond_1_adam_iter_read_readvariableop	1
-savev2_current_loss_scale_read_readvariableop)
%savev2_good_steps_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop#
savev2_aucs_read_readvariableop!
savev2_ns_read_readvariableopD
@savev2_cond_1_adam_extract_features_kernel_m_read_readvariableopB
>savev2_cond_1_adam_extract_features_bias_m_read_readvariableopF
Bsavev2_cond_1_adam_extract_features_bn_gamma_m_read_readvariableopE
Asavev2_cond_1_adam_extract_features_bn_beta_m_read_readvariableop9
5savev2_cond_1_adam_conv1_kernel_m_read_readvariableop7
3savev2_cond_1_adam_conv1_bias_m_read_readvariableop;
7savev2_cond_1_adam_conv1_bn_gamma_m_read_readvariableop:
6savev2_cond_1_adam_conv1_bn_beta_m_read_readvariableop9
5savev2_cond_1_adam_conv2_kernel_m_read_readvariableop7
3savev2_cond_1_adam_conv2_bias_m_read_readvariableop;
7savev2_cond_1_adam_conv2_bn_gamma_m_read_readvariableop:
6savev2_cond_1_adam_conv2_bn_beta_m_read_readvariableop<
8savev2_cond_1_adam_d1_dense_kernel_m_read_readvariableop:
6savev2_cond_1_adam_d1_dense_bias_m_read_readvariableop8
4savev2_cond_1_adam_d1_bn_gamma_m_read_readvariableop7
3savev2_cond_1_adam_d1_bn_beta_m_read_readvariableop<
8savev2_cond_1_adam_d2_dense_kernel_m_read_readvariableop:
6savev2_cond_1_adam_d2_dense_bias_m_read_readvariableop8
4savev2_cond_1_adam_d2_bn_gamma_m_read_readvariableop7
3savev2_cond_1_adam_d2_bn_beta_m_read_readvariableopA
=savev2_cond_1_adam_dense_predict_kernel_m_read_readvariableop?
;savev2_cond_1_adam_dense_predict_bias_m_read_readvariableopD
@savev2_cond_1_adam_extract_features_kernel_v_read_readvariableopB
>savev2_cond_1_adam_extract_features_bias_v_read_readvariableopF
Bsavev2_cond_1_adam_extract_features_bn_gamma_v_read_readvariableopE
Asavev2_cond_1_adam_extract_features_bn_beta_v_read_readvariableop9
5savev2_cond_1_adam_conv1_kernel_v_read_readvariableop7
3savev2_cond_1_adam_conv1_bias_v_read_readvariableop;
7savev2_cond_1_adam_conv1_bn_gamma_v_read_readvariableop:
6savev2_cond_1_adam_conv1_bn_beta_v_read_readvariableop9
5savev2_cond_1_adam_conv2_kernel_v_read_readvariableop7
3savev2_cond_1_adam_conv2_bias_v_read_readvariableop;
7savev2_cond_1_adam_conv2_bn_gamma_v_read_readvariableop:
6savev2_cond_1_adam_conv2_bn_beta_v_read_readvariableop<
8savev2_cond_1_adam_d1_dense_kernel_v_read_readvariableop:
6savev2_cond_1_adam_d1_dense_bias_v_read_readvariableop8
4savev2_cond_1_adam_d1_bn_gamma_v_read_readvariableop7
3savev2_cond_1_adam_d1_bn_beta_v_read_readvariableop<
8savev2_cond_1_adam_d2_dense_kernel_v_read_readvariableop:
6savev2_cond_1_adam_d2_dense_bias_v_read_readvariableop8
4savev2_cond_1_adam_d2_bn_gamma_v_read_readvariableop7
3savev2_cond_1_adam_d2_bn_beta_v_read_readvariableopA
=savev2_cond_1_adam_dense_predict_kernel_v_read_readvariableop?
;savev2_cond_1_adam_dense_predict_bias_v_read_readvariableop
savev2_const_1

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ®0
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*×/
valueÍ/BÊ/XB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEBBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB3keras_api/metrics/1/AUCs/.ATTRIBUTES/VARIABLE_VALUEB1keras_api/metrics/1/Ns/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH 
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*Å
value»B¸XB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B á$
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_extract_features_kernel_read_readvariableop0savev2_extract_features_bias_read_readvariableop4savev2_extract_features_bn_gamma_read_readvariableop3savev2_extract_features_bn_beta_read_readvariableop:savev2_extract_features_bn_moving_mean_read_readvariableop>savev2_extract_features_bn_moving_variance_read_readvariableop'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop)savev2_conv1_bn_gamma_read_readvariableop(savev2_conv1_bn_beta_read_readvariableop/savev2_conv1_bn_moving_mean_read_readvariableop3savev2_conv1_bn_moving_variance_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop)savev2_conv2_bn_gamma_read_readvariableop(savev2_conv2_bn_beta_read_readvariableop/savev2_conv2_bn_moving_mean_read_readvariableop3savev2_conv2_bn_moving_variance_read_readvariableop*savev2_d1_dense_kernel_read_readvariableop(savev2_d1_dense_bias_read_readvariableop&savev2_d1_bn_gamma_read_readvariableop%savev2_d1_bn_beta_read_readvariableop,savev2_d1_bn_moving_mean_read_readvariableop0savev2_d1_bn_moving_variance_read_readvariableop*savev2_d2_dense_kernel_read_readvariableop(savev2_d2_dense_bias_read_readvariableop&savev2_d2_bn_gamma_read_readvariableop%savev2_d2_bn_beta_read_readvariableop,savev2_d2_bn_moving_mean_read_readvariableop0savev2_d2_bn_moving_variance_read_readvariableop/savev2_dense_predict_kernel_read_readvariableop-savev2_dense_predict_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop+savev2_cond_1_adam_iter_read_readvariableop-savev2_current_loss_scale_read_readvariableop%savev2_good_steps_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_aucs_read_readvariableopsavev2_ns_read_readvariableop@savev2_cond_1_adam_extract_features_kernel_m_read_readvariableop>savev2_cond_1_adam_extract_features_bias_m_read_readvariableopBsavev2_cond_1_adam_extract_features_bn_gamma_m_read_readvariableopAsavev2_cond_1_adam_extract_features_bn_beta_m_read_readvariableop5savev2_cond_1_adam_conv1_kernel_m_read_readvariableop3savev2_cond_1_adam_conv1_bias_m_read_readvariableop7savev2_cond_1_adam_conv1_bn_gamma_m_read_readvariableop6savev2_cond_1_adam_conv1_bn_beta_m_read_readvariableop5savev2_cond_1_adam_conv2_kernel_m_read_readvariableop3savev2_cond_1_adam_conv2_bias_m_read_readvariableop7savev2_cond_1_adam_conv2_bn_gamma_m_read_readvariableop6savev2_cond_1_adam_conv2_bn_beta_m_read_readvariableop8savev2_cond_1_adam_d1_dense_kernel_m_read_readvariableop6savev2_cond_1_adam_d1_dense_bias_m_read_readvariableop4savev2_cond_1_adam_d1_bn_gamma_m_read_readvariableop3savev2_cond_1_adam_d1_bn_beta_m_read_readvariableop8savev2_cond_1_adam_d2_dense_kernel_m_read_readvariableop6savev2_cond_1_adam_d2_dense_bias_m_read_readvariableop4savev2_cond_1_adam_d2_bn_gamma_m_read_readvariableop3savev2_cond_1_adam_d2_bn_beta_m_read_readvariableop=savev2_cond_1_adam_dense_predict_kernel_m_read_readvariableop;savev2_cond_1_adam_dense_predict_bias_m_read_readvariableop@savev2_cond_1_adam_extract_features_kernel_v_read_readvariableop>savev2_cond_1_adam_extract_features_bias_v_read_readvariableopBsavev2_cond_1_adam_extract_features_bn_gamma_v_read_readvariableopAsavev2_cond_1_adam_extract_features_bn_beta_v_read_readvariableop5savev2_cond_1_adam_conv1_kernel_v_read_readvariableop3savev2_cond_1_adam_conv1_bias_v_read_readvariableop7savev2_cond_1_adam_conv1_bn_gamma_v_read_readvariableop6savev2_cond_1_adam_conv1_bn_beta_v_read_readvariableop5savev2_cond_1_adam_conv2_kernel_v_read_readvariableop3savev2_cond_1_adam_conv2_bias_v_read_readvariableop7savev2_cond_1_adam_conv2_bn_gamma_v_read_readvariableop6savev2_cond_1_adam_conv2_bn_beta_v_read_readvariableop8savev2_cond_1_adam_d1_dense_kernel_v_read_readvariableop6savev2_cond_1_adam_d1_dense_bias_v_read_readvariableop4savev2_cond_1_adam_d1_bn_gamma_v_read_readvariableop3savev2_cond_1_adam_d1_bn_beta_v_read_readvariableop8savev2_cond_1_adam_d2_dense_kernel_v_read_readvariableop6savev2_cond_1_adam_d2_dense_bias_v_read_readvariableop4savev2_cond_1_adam_d2_bn_gamma_v_read_readvariableop3savev2_cond_1_adam_d2_bn_beta_v_read_readvariableop=savev2_cond_1_adam_dense_predict_kernel_v_read_readvariableop;savev2_cond_1_adam_dense_predict_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *f
dtypes\
Z2X		
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesû
ø: :@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:	@::::::
::::::	:: : : : : : : : : :::@:@:@:@:@@:@:@:@:@@:@:@:@:	@::::
::::	::@:@:@:@:@@:@:@:@:@@:@:@:@:	@::::
::::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:	:  

_output_shapes
::!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: : *

_output_shapes
:: +

_output_shapes
::(,$
"
_output_shapes
:@: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@:(0$
"
_output_shapes
:@@: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@:(4$
"
_output_shapes
:@@: 5

_output_shapes
:@: 6

_output_shapes
:@: 7

_output_shapes
:@:%8!

_output_shapes
:	@:!9

_output_shapes	
::!:

_output_shapes	
::!;

_output_shapes	
::&<"
 
_output_shapes
:
:!=

_output_shapes	
::!>

_output_shapes	
::!?

_output_shapes	
::%@!

_output_shapes
:	: A

_output_shapes
::(B$
"
_output_shapes
:@: C

_output_shapes
:@: D

_output_shapes
:@: E

_output_shapes
:@:(F$
"
_output_shapes
:@@: G

_output_shapes
:@: H

_output_shapes
:@: I

_output_shapes
:@:(J$
"
_output_shapes
:@@: K

_output_shapes
:@: L

_output_shapes
:@: M

_output_shapes
:@:%N!

_output_shapes
:	@:!O

_output_shapes	
::!P

_output_shapes	
::!Q

_output_shapes	
::&R"
 
_output_shapes
:
:!S

_output_shapes	
::!T

_output_shapes	
::!U

_output_shapes	
::%V!

_output_shapes
:	: W

_output_shapes
::X

_output_shapes
: 
ü	
e
F__inference_d1_dropout_layer_call_and_return_conditional_losses_559429

inputs
identityP
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B jzd
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seed{Y
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B jæd¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ü	
e
F__inference_d1_dropout_layer_call_and_return_conditional_losses_557699

inputs
identityP
dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B jzd
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seed{Y
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B jæd¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@Y
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¤
A__inference_d2_BN_layer_call_and_return_conditional_losses_557223

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:f
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity
Cast_1:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
K
__inference_loss_12088

y_true

y_pred
unknown
identity®
PartitionedCallPartitionedCally_truey_predunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *:
f5R3
1__inference_weighted_masked_BCE_from_logits_12083\
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namey_true:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namey_pred: 

_output_shapes
:
 
e
I__inference_conv2_dropout_layer_call_and_return_conditional_losses_559234

inputs
identity[
IdentityIdentityinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 
e
I__inference_conv1_dropout_layer_call_and_return_conditional_losses_557776

inputs
identity[
IdentityIdentityinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ö
d
+__inference_d1_dropout_layer_call_fn_559412

inputs
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_d1_dropout_layer_call_and_return_conditional_losses_557699o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

Å
&__inference_d1_BN_layer_call_fn_559463

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_d1_BN_layer_call_and_return_conditional_losses_557137p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
Ä
&__inference_model_layer_call_fn_558427

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	@

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:


unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_557547o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð	
û
I__inference_dense_predict_layer_call_and_return_conditional_losses_557540

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
Ä
)__inference_conv2_BN_layer_call_fn_559286

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2_BN_layer_call_and_return_conditional_losses_557072|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

÷
D__inference_d1_dense_layer_call_and_return_conditional_losses_559450

inputs1
matmul_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0k
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	@\
MatMulMatMulinputsMatMul/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:i
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

`
D__inference_conv1_mp_layer_call_and_return_conditional_losses_557383

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :|

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¦
MaxPoolMaxPoolExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
z
SqueezeSqueezeMaxPool:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
e
IdentityIdentitySqueeze:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ä'
Ý
D__inference_conv1_BN_layer_call_and_return_conditional_losses_559179

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpb
CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@o
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ¢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@r
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@q
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
IdentityIdentity
Cast_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò

.__inference_dense_predict_layer_call_fn_559695

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_dense_predict_layer_call_and_return_conditional_losses_557540o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
®
O__inference_extract_features_BN_layer_call_and_return_conditional_losses_559004

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpb
CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@r
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@q
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
IdentityIdentity
Cast_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ê
×
!__inference__wrapped_model_556810	
inputX
Bmodel_extract_features_conv1d_expanddims_1_readvariableop_resource:@D
6model_extract_features_biasadd_readvariableop_resource:@I
;model_extract_features_bn_batchnorm_readvariableop_resource:@M
?model_extract_features_bn_batchnorm_mul_readvariableop_resource:@K
=model_extract_features_bn_batchnorm_readvariableop_1_resource:@K
=model_extract_features_bn_batchnorm_readvariableop_2_resource:@M
7model_conv1_conv1d_expanddims_1_readvariableop_resource:@@9
+model_conv1_biasadd_readvariableop_resource:@>
0model_conv1_bn_batchnorm_readvariableop_resource:@B
4model_conv1_bn_batchnorm_mul_readvariableop_resource:@@
2model_conv1_bn_batchnorm_readvariableop_1_resource:@@
2model_conv1_bn_batchnorm_readvariableop_2_resource:@M
7model_conv2_conv1d_expanddims_1_readvariableop_resource:@@9
+model_conv2_biasadd_readvariableop_resource:@>
0model_conv2_bn_batchnorm_readvariableop_resource:@B
4model_conv2_bn_batchnorm_mul_readvariableop_resource:@@
2model_conv2_bn_batchnorm_readvariableop_1_resource:@@
2model_conv2_bn_batchnorm_readvariableop_2_resource:@@
-model_d1_dense_matmul_readvariableop_resource:	@=
.model_d1_dense_biasadd_readvariableop_resource:	<
-model_d1_bn_batchnorm_readvariableop_resource:	@
1model_d1_bn_batchnorm_mul_readvariableop_resource:	>
/model_d1_bn_batchnorm_readvariableop_1_resource:	>
/model_d1_bn_batchnorm_readvariableop_2_resource:	A
-model_d2_dense_matmul_readvariableop_resource:
=
.model_d2_dense_biasadd_readvariableop_resource:	<
-model_d2_bn_batchnorm_readvariableop_resource:	@
1model_d2_bn_batchnorm_mul_readvariableop_resource:	>
/model_d2_bn_batchnorm_readvariableop_1_resource:	>
/model_d2_bn_batchnorm_readvariableop_2_resource:	E
2model_dense_predict_matmul_readvariableop_resource:	A
3model_dense_predict_biasadd_readvariableop_resource:
identity¢"model/conv1/BiasAdd/ReadVariableOp¢.model/conv1/Conv1D/ExpandDims_1/ReadVariableOp¢'model/conv1_BN/batchnorm/ReadVariableOp¢)model/conv1_BN/batchnorm/ReadVariableOp_1¢)model/conv1_BN/batchnorm/ReadVariableOp_2¢+model/conv1_BN/batchnorm/mul/ReadVariableOp¢"model/conv2/BiasAdd/ReadVariableOp¢.model/conv2/Conv1D/ExpandDims_1/ReadVariableOp¢'model/conv2_BN/batchnorm/ReadVariableOp¢)model/conv2_BN/batchnorm/ReadVariableOp_1¢)model/conv2_BN/batchnorm/ReadVariableOp_2¢+model/conv2_BN/batchnorm/mul/ReadVariableOp¢$model/d1_BN/batchnorm/ReadVariableOp¢&model/d1_BN/batchnorm/ReadVariableOp_1¢&model/d1_BN/batchnorm/ReadVariableOp_2¢(model/d1_BN/batchnorm/mul/ReadVariableOp¢%model/d1_dense/BiasAdd/ReadVariableOp¢$model/d1_dense/MatMul/ReadVariableOp¢$model/d2_BN/batchnorm/ReadVariableOp¢&model/d2_BN/batchnorm/ReadVariableOp_1¢&model/d2_BN/batchnorm/ReadVariableOp_2¢(model/d2_BN/batchnorm/mul/ReadVariableOp¢%model/d2_dense/BiasAdd/ReadVariableOp¢$model/d2_dense/MatMul/ReadVariableOp¢*model/dense_predict/BiasAdd/ReadVariableOp¢)model/dense_predict/MatMul/ReadVariableOp¢-model/extract_features/BiasAdd/ReadVariableOp¢9model/extract_features/Conv1D/ExpandDims_1/ReadVariableOp¢2model/extract_features_BN/batchnorm/ReadVariableOp¢4model/extract_features_BN/batchnorm/ReadVariableOp_1¢4model/extract_features_BN/batchnorm/ReadVariableOp_2¢6model/extract_features_BN/batchnorm/mul/ReadVariableOpx
model/extract_features/CastCastinput*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿw
,model/extract_features/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÑ
(model/extract_features/Conv1D/ExpandDims
ExpandDimsmodel/extract_features/Cast:y:05model/extract_features/Conv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
9model/extract_features/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpBmodel_extract_features_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0¶
/model/extract_features/Conv1D/ExpandDims_1/CastCastAmodel/extract_features/Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@p
.model/extract_features/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ×
*model/extract_features/Conv1D/ExpandDims_1
ExpandDims3model/extract_features/Conv1D/ExpandDims_1/Cast:y:07model/extract_features/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@û
model/extract_features/Conv1DConv2D1model/extract_features/Conv1D/ExpandDims:output:03model/extract_features/Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
·
%model/extract_features/Conv1D/SqueezeSqueeze&model/extract_features/Conv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ 
-model/extract_features/BiasAdd/ReadVariableOpReadVariableOp6model_extract_features_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
#model/extract_features/BiasAdd/CastCast5model/extract_features/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@Á
model/extract_features/BiasAddBiasAdd.model/extract_features/Conv1D/Squeeze:output:0'model/extract_features/BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
model/extract_features_BN/CastCast'model/extract_features/BiasAdd:output:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ª
2model/extract_features_BN/batchnorm/ReadVariableOpReadVariableOp;model_extract_features_bn_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0n
)model/extract_features_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Å
'model/extract_features_BN/batchnorm/addAddV2:model/extract_features_BN/batchnorm/ReadVariableOp:value:02model/extract_features_BN/batchnorm/add/y:output:0*
T0*
_output_shapes
:@
)model/extract_features_BN/batchnorm/RsqrtRsqrt+model/extract_features_BN/batchnorm/add:z:0*
T0*
_output_shapes
:@²
6model/extract_features_BN/batchnorm/mul/ReadVariableOpReadVariableOp?model_extract_features_bn_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Â
'model/extract_features_BN/batchnorm/mulMul-model/extract_features_BN/batchnorm/Rsqrt:y:0>model/extract_features_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@À
)model/extract_features_BN/batchnorm/mul_1Mul"model/extract_features_BN/Cast:y:0+model/extract_features_BN/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@®
4model/extract_features_BN/batchnorm/ReadVariableOp_1ReadVariableOp=model_extract_features_bn_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0À
)model/extract_features_BN/batchnorm/mul_2Mul<model/extract_features_BN/batchnorm/ReadVariableOp_1:value:0+model/extract_features_BN/batchnorm/mul:z:0*
T0*
_output_shapes
:@®
4model/extract_features_BN/batchnorm/ReadVariableOp_2ReadVariableOp=model_extract_features_bn_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0À
'model/extract_features_BN/batchnorm/subSub<model/extract_features_BN/batchnorm/ReadVariableOp_2:value:0-model/extract_features_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@Í
)model/extract_features_BN/batchnorm/add_1AddV2-model/extract_features_BN/batchnorm/mul_1:z:0+model/extract_features_BN/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¥
 model/extract_features_BN/Cast_1Cast-model/extract_features_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 model/extract_features_RELU/ReluRelu$model/extract_features_BN/Cast_1:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
model/conv1_dropout/IdentityIdentity.model/extract_features_RELU/Relu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@l
!model/conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÁ
model/conv1/Conv1D/ExpandDims
ExpandDims%model/conv1_dropout/Identity:output:0*model/conv1/Conv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ª
.model/conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp7model_conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0 
$model/conv1/Conv1D/ExpandDims_1/CastCast6model/conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@@e
#model/conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¶
model/conv1/Conv1D/ExpandDims_1
ExpandDims(model/conv1/Conv1D/ExpandDims_1/Cast:y:0,model/conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@Ú
model/conv1/Conv1DConv2D&model/conv1/Conv1D/ExpandDims:output:0(model/conv1/Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¡
model/conv1/Conv1D/SqueezeSqueezemodel/conv1/Conv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
"model/conv1/BiasAdd/ReadVariableOpReadVariableOp+model_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
model/conv1/BiasAdd/CastCast*model/conv1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@ 
model/conv1/BiasAddBiasAdd#model/conv1/Conv1D/Squeeze:output:0model/conv1/BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
model/conv1_BN/CastCastmodel/conv1/BiasAdd:output:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
'model/conv1_BN/batchnorm/ReadVariableOpReadVariableOp0model_conv1_bn_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0c
model/conv1_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¤
model/conv1_BN/batchnorm/addAddV2/model/conv1_BN/batchnorm/ReadVariableOp:value:0'model/conv1_BN/batchnorm/add/y:output:0*
T0*
_output_shapes
:@n
model/conv1_BN/batchnorm/RsqrtRsqrt model/conv1_BN/batchnorm/add:z:0*
T0*
_output_shapes
:@
+model/conv1_BN/batchnorm/mul/ReadVariableOpReadVariableOp4model_conv1_bn_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0¡
model/conv1_BN/batchnorm/mulMul"model/conv1_BN/batchnorm/Rsqrt:y:03model/conv1_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@
model/conv1_BN/batchnorm/mul_1Mulmodel/conv1_BN/Cast:y:0 model/conv1_BN/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
)model/conv1_BN/batchnorm/ReadVariableOp_1ReadVariableOp2model_conv1_bn_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0
model/conv1_BN/batchnorm/mul_2Mul1model/conv1_BN/batchnorm/ReadVariableOp_1:value:0 model/conv1_BN/batchnorm/mul:z:0*
T0*
_output_shapes
:@
)model/conv1_BN/batchnorm/ReadVariableOp_2ReadVariableOp2model_conv1_bn_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0
model/conv1_BN/batchnorm/subSub1model/conv1_BN/batchnorm/ReadVariableOp_2:value:0"model/conv1_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@¬
model/conv1_BN/batchnorm/add_1AddV2"model/conv1_BN/batchnorm/mul_1:z:0 model/conv1_BN/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
model/conv1_BN/Cast_1Cast"model/conv1_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@w
model/conv1_RELU/ReluRelumodel/conv1_BN/Cast_1:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@_
model/conv1_mp/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :·
model/conv1_mp/ExpandDims
ExpandDims#model/conv1_RELU/Relu:activations:0&model/conv1_mp/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ä
model/conv1_mp/MaxPoolMaxPool"model/conv1_mp/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

model/conv1_mp/SqueezeSqueezemodel/conv1_mp/MaxPool:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

model/conv2_dropout/IdentityIdentitymodel/conv1_mp/Squeeze:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@l
!model/conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿÁ
model/conv2/Conv1D/ExpandDims
ExpandDims%model/conv2_dropout/Identity:output:0*model/conv2/Conv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ª
.model/conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp7model_conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0 
$model/conv2/Conv1D/ExpandDims_1/CastCast6model/conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@@e
#model/conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¶
model/conv2/Conv1D/ExpandDims_1
ExpandDims(model/conv2/Conv1D/ExpandDims_1/Cast:y:0,model/conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@Ú
model/conv2/Conv1DConv2D&model/conv2/Conv1D/ExpandDims:output:0(model/conv2/Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
¡
model/conv2/Conv1D/SqueezeSqueezemodel/conv2/Conv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
"model/conv2/BiasAdd/ReadVariableOpReadVariableOp+model_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
model/conv2/BiasAdd/CastCast*model/conv2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@ 
model/conv2/BiasAddBiasAdd#model/conv2/Conv1D/Squeeze:output:0model/conv2/BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
model/conv2_BN/CastCastmodel/conv2/BiasAdd:output:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
'model/conv2_BN/batchnorm/ReadVariableOpReadVariableOp0model_conv2_bn_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0c
model/conv2_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¤
model/conv2_BN/batchnorm/addAddV2/model/conv2_BN/batchnorm/ReadVariableOp:value:0'model/conv2_BN/batchnorm/add/y:output:0*
T0*
_output_shapes
:@n
model/conv2_BN/batchnorm/RsqrtRsqrt model/conv2_BN/batchnorm/add:z:0*
T0*
_output_shapes
:@
+model/conv2_BN/batchnorm/mul/ReadVariableOpReadVariableOp4model_conv2_bn_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0¡
model/conv2_BN/batchnorm/mulMul"model/conv2_BN/batchnorm/Rsqrt:y:03model/conv2_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@
model/conv2_BN/batchnorm/mul_1Mulmodel/conv2_BN/Cast:y:0 model/conv2_BN/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
)model/conv2_BN/batchnorm/ReadVariableOp_1ReadVariableOp2model_conv2_bn_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0
model/conv2_BN/batchnorm/mul_2Mul1model/conv2_BN/batchnorm/ReadVariableOp_1:value:0 model/conv2_BN/batchnorm/mul:z:0*
T0*
_output_shapes
:@
)model/conv2_BN/batchnorm/ReadVariableOp_2ReadVariableOp2model_conv2_bn_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0
model/conv2_BN/batchnorm/subSub1model/conv2_BN/batchnorm/ReadVariableOp_2:value:0"model/conv2_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@¬
model/conv2_BN/batchnorm/add_1AddV2"model/conv2_BN/batchnorm/mul_1:z:0 model/conv2_BN/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
model/conv2_BN/Cast_1Cast"model/conv2_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@w
model/conv2_RELU/ReluRelumodel/conv2_BN/Cast_1:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@_
model/conv2_mp/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :·
model/conv2_mp/ExpandDims
ExpandDims#model/conv2_RELU/Relu:activations:0&model/conv2_mp/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ä
model/conv2_mp/MaxPoolMaxPool"model/conv2_mp/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

model/conv2_mp/SqueezeSqueezemodel/conv2_mp/MaxPool:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
n
,model/combine_features/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :«
model/combine_features/MaxMaxmodel/conv2_mp/Squeeze:output:05model/combine_features/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
model/d1_dropout/IdentityIdentity#model/combine_features/Max:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$model/d1_dense/MatMul/ReadVariableOpReadVariableOp-model_d1_dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
model/d1_dense/MatMul/CastCast,model/d1_dense/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	@
model/d1_dense/MatMulMatMul"model/d1_dropout/Identity:output:0model/d1_dense/MatMul/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%model/d1_dense/BiasAdd/ReadVariableOpReadVariableOp.model_d1_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
model/d1_dense/BiasAdd/CastCast-model/d1_dense/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:
model/d1_dense/BiasAddBiasAddmodel/d1_dense/MatMul:product:0model/d1_dense/BiasAdd/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
model/d1_BN/CastCastmodel/d1_dense/BiasAdd:output:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/d1_BN/batchnorm/ReadVariableOpReadVariableOp-model_d1_bn_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0`
model/d1_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
model/d1_BN/batchnorm/addAddV2,model/d1_BN/batchnorm/ReadVariableOp:value:0$model/d1_BN/batchnorm/add/y:output:0*
T0*
_output_shapes	
:i
model/d1_BN/batchnorm/RsqrtRsqrtmodel/d1_BN/batchnorm/add:z:0*
T0*
_output_shapes	
:
(model/d1_BN/batchnorm/mul/ReadVariableOpReadVariableOp1model_d1_bn_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0
model/d1_BN/batchnorm/mulMulmodel/d1_BN/batchnorm/Rsqrt:y:00model/d1_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:
model/d1_BN/batchnorm/mul_1Mulmodel/d1_BN/Cast:y:0model/d1_BN/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model/d1_BN/batchnorm/ReadVariableOp_1ReadVariableOp/model_d1_bn_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0
model/d1_BN/batchnorm/mul_2Mul.model/d1_BN/batchnorm/ReadVariableOp_1:value:0model/d1_BN/batchnorm/mul:z:0*
T0*
_output_shapes	
:
&model/d1_BN/batchnorm/ReadVariableOp_2ReadVariableOp/model_d1_bn_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0
model/d1_BN/batchnorm/subSub.model/d1_BN/batchnorm/ReadVariableOp_2:value:0model/d1_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
model/d1_BN/batchnorm/add_1AddV2model/d1_BN/batchnorm/mul_1:z:0model/d1_BN/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
model/d1_BN/Cast_1Castmodel/d1_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
model/d1_RELU/ReluRelumodel/d1_BN/Cast_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
model/d2_dropout/IdentityIdentity model/d1_RELU/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/d2_dense/MatMul/ReadVariableOpReadVariableOp-model_d2_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
model/d2_dense/MatMul/CastCast,model/d2_dense/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:

model/d2_dense/MatMulMatMul"model/d2_dropout/Identity:output:0model/d2_dense/MatMul/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%model/d2_dense/BiasAdd/ReadVariableOpReadVariableOp.model_d2_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
model/d2_dense/BiasAdd/CastCast-model/d2_dense/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:
model/d2_dense/BiasAddBiasAddmodel/d2_dense/MatMul:product:0model/d2_dense/BiasAdd/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
model/d2_BN/CastCastmodel/d2_dense/BiasAdd:output:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/d2_BN/batchnorm/ReadVariableOpReadVariableOp-model_d2_bn_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0`
model/d2_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
model/d2_BN/batchnorm/addAddV2,model/d2_BN/batchnorm/ReadVariableOp:value:0$model/d2_BN/batchnorm/add/y:output:0*
T0*
_output_shapes	
:i
model/d2_BN/batchnorm/RsqrtRsqrtmodel/d2_BN/batchnorm/add:z:0*
T0*
_output_shapes	
:
(model/d2_BN/batchnorm/mul/ReadVariableOpReadVariableOp1model_d2_bn_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0
model/d2_BN/batchnorm/mulMulmodel/d2_BN/batchnorm/Rsqrt:y:00model/d2_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:
model/d2_BN/batchnorm/mul_1Mulmodel/d2_BN/Cast:y:0model/d2_BN/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model/d2_BN/batchnorm/ReadVariableOp_1ReadVariableOp/model_d2_bn_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0
model/d2_BN/batchnorm/mul_2Mul.model/d2_BN/batchnorm/ReadVariableOp_1:value:0model/d2_BN/batchnorm/mul:z:0*
T0*
_output_shapes	
:
&model/d2_BN/batchnorm/ReadVariableOp_2ReadVariableOp/model_d2_bn_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0
model/d2_BN/batchnorm/subSub.model/d2_BN/batchnorm/ReadVariableOp_2:value:0model/d2_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
model/d2_BN/batchnorm/add_1AddV2model/d2_BN/batchnorm/mul_1:z:0model/d2_BN/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
model/d2_BN/Cast_1Castmodel/d2_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
model/d2_RELU/ReluRelumodel/d2_BN/Cast_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/dense_predict/CastCast model/d2_RELU/Relu:activations:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model/dense_predict/MatMul/ReadVariableOpReadVariableOp2model_dense_predict_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0§
model/dense_predict/MatMulMatMulmodel/dense_predict/Cast:y:01model/dense_predict/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*model/dense_predict/BiasAdd/ReadVariableOpReadVariableOp3model_dense_predict_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0²
model/dense_predict/BiasAddBiasAdd$model/dense_predict/MatMul:product:02model/dense_predict/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
IdentityIdentity$model/dense_predict/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
NoOpNoOp#^model/conv1/BiasAdd/ReadVariableOp/^model/conv1/Conv1D/ExpandDims_1/ReadVariableOp(^model/conv1_BN/batchnorm/ReadVariableOp*^model/conv1_BN/batchnorm/ReadVariableOp_1*^model/conv1_BN/batchnorm/ReadVariableOp_2,^model/conv1_BN/batchnorm/mul/ReadVariableOp#^model/conv2/BiasAdd/ReadVariableOp/^model/conv2/Conv1D/ExpandDims_1/ReadVariableOp(^model/conv2_BN/batchnorm/ReadVariableOp*^model/conv2_BN/batchnorm/ReadVariableOp_1*^model/conv2_BN/batchnorm/ReadVariableOp_2,^model/conv2_BN/batchnorm/mul/ReadVariableOp%^model/d1_BN/batchnorm/ReadVariableOp'^model/d1_BN/batchnorm/ReadVariableOp_1'^model/d1_BN/batchnorm/ReadVariableOp_2)^model/d1_BN/batchnorm/mul/ReadVariableOp&^model/d1_dense/BiasAdd/ReadVariableOp%^model/d1_dense/MatMul/ReadVariableOp%^model/d2_BN/batchnorm/ReadVariableOp'^model/d2_BN/batchnorm/ReadVariableOp_1'^model/d2_BN/batchnorm/ReadVariableOp_2)^model/d2_BN/batchnorm/mul/ReadVariableOp&^model/d2_dense/BiasAdd/ReadVariableOp%^model/d2_dense/MatMul/ReadVariableOp+^model/dense_predict/BiasAdd/ReadVariableOp*^model/dense_predict/MatMul/ReadVariableOp.^model/extract_features/BiasAdd/ReadVariableOp:^model/extract_features/Conv1D/ExpandDims_1/ReadVariableOp3^model/extract_features_BN/batchnorm/ReadVariableOp5^model/extract_features_BN/batchnorm/ReadVariableOp_15^model/extract_features_BN/batchnorm/ReadVariableOp_27^model/extract_features_BN/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"model/conv1/BiasAdd/ReadVariableOp"model/conv1/BiasAdd/ReadVariableOp2`
.model/conv1/Conv1D/ExpandDims_1/ReadVariableOp.model/conv1/Conv1D/ExpandDims_1/ReadVariableOp2R
'model/conv1_BN/batchnorm/ReadVariableOp'model/conv1_BN/batchnorm/ReadVariableOp2V
)model/conv1_BN/batchnorm/ReadVariableOp_1)model/conv1_BN/batchnorm/ReadVariableOp_12V
)model/conv1_BN/batchnorm/ReadVariableOp_2)model/conv1_BN/batchnorm/ReadVariableOp_22Z
+model/conv1_BN/batchnorm/mul/ReadVariableOp+model/conv1_BN/batchnorm/mul/ReadVariableOp2H
"model/conv2/BiasAdd/ReadVariableOp"model/conv2/BiasAdd/ReadVariableOp2`
.model/conv2/Conv1D/ExpandDims_1/ReadVariableOp.model/conv2/Conv1D/ExpandDims_1/ReadVariableOp2R
'model/conv2_BN/batchnorm/ReadVariableOp'model/conv2_BN/batchnorm/ReadVariableOp2V
)model/conv2_BN/batchnorm/ReadVariableOp_1)model/conv2_BN/batchnorm/ReadVariableOp_12V
)model/conv2_BN/batchnorm/ReadVariableOp_2)model/conv2_BN/batchnorm/ReadVariableOp_22Z
+model/conv2_BN/batchnorm/mul/ReadVariableOp+model/conv2_BN/batchnorm/mul/ReadVariableOp2L
$model/d1_BN/batchnorm/ReadVariableOp$model/d1_BN/batchnorm/ReadVariableOp2P
&model/d1_BN/batchnorm/ReadVariableOp_1&model/d1_BN/batchnorm/ReadVariableOp_12P
&model/d1_BN/batchnorm/ReadVariableOp_2&model/d1_BN/batchnorm/ReadVariableOp_22T
(model/d1_BN/batchnorm/mul/ReadVariableOp(model/d1_BN/batchnorm/mul/ReadVariableOp2N
%model/d1_dense/BiasAdd/ReadVariableOp%model/d1_dense/BiasAdd/ReadVariableOp2L
$model/d1_dense/MatMul/ReadVariableOp$model/d1_dense/MatMul/ReadVariableOp2L
$model/d2_BN/batchnorm/ReadVariableOp$model/d2_BN/batchnorm/ReadVariableOp2P
&model/d2_BN/batchnorm/ReadVariableOp_1&model/d2_BN/batchnorm/ReadVariableOp_12P
&model/d2_BN/batchnorm/ReadVariableOp_2&model/d2_BN/batchnorm/ReadVariableOp_22T
(model/d2_BN/batchnorm/mul/ReadVariableOp(model/d2_BN/batchnorm/mul/ReadVariableOp2N
%model/d2_dense/BiasAdd/ReadVariableOp%model/d2_dense/BiasAdd/ReadVariableOp2L
$model/d2_dense/MatMul/ReadVariableOp$model/d2_dense/MatMul/ReadVariableOp2X
*model/dense_predict/BiasAdd/ReadVariableOp*model/dense_predict/BiasAdd/ReadVariableOp2V
)model/dense_predict/MatMul/ReadVariableOp)model/dense_predict/MatMul/ReadVariableOp2^
-model/extract_features/BiasAdd/ReadVariableOp-model/extract_features/BiasAdd/ReadVariableOp2v
9model/extract_features/Conv1D/ExpandDims_1/ReadVariableOp9model/extract_features/Conv1D/ExpandDims_1/ReadVariableOp2h
2model/extract_features_BN/batchnorm/ReadVariableOp2model/extract_features_BN/batchnorm/ReadVariableOp2l
4model/extract_features_BN/batchnorm/ReadVariableOp_14model/extract_features_BN/batchnorm/ReadVariableOp_12l
4model/extract_features_BN/batchnorm/ReadVariableOp_24model/extract_features_BN/batchnorm/ReadVariableOp_22p
6model/extract_features_BN/batchnorm/mul/ReadVariableOp6model/extract_features_BN/batchnorm/mul/ReadVariableOp:[ W
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
Ò
`
D__inference_conv2_mp_layer_call_and_return_conditional_losses_557095

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¯
MaxPoolMaxPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


A__inference_conv1_layer_call_and_return_conditional_losses_557354

inputsA
+conv1d_expanddims_1_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
Conv1D/ExpandDims_1/CastCast*Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@@Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Conv1D/ExpandDims_1
ExpandDimsConv1D/ExpandDims_1/Cast:y:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@¶
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@|
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

g
I__inference_conv2_dropout_layer_call_and_return_conditional_losses_559230

inputs

identity_1[
IdentityIdentityinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
J
.__inference_conv1_dropout_layer_call_fn_559060

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv1_dropout_layer_call_and_return_conditional_losses_557776m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ø
E
)__inference_conv2_mp_layer_call_fn_559359

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2_mp_layer_call_and_return_conditional_losses_557095v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
`
D__inference_conv2_mp_layer_call_and_return_conditional_losses_559372

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¯
MaxPoolMaxPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä'
Ý
D__inference_conv2_BN_layer_call_and_return_conditional_losses_559344

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpb
CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@o
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ¢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@r
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@q
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
IdentityIdentity
Cast_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ü
Î7
"__inference__traced_restore_560261
file_prefix>
(assignvariableop_extract_features_kernel:@6
(assignvariableop_1_extract_features_bias:@:
,assignvariableop_2_extract_features_bn_gamma:@9
+assignvariableop_3_extract_features_bn_beta:@@
2assignvariableop_4_extract_features_bn_moving_mean:@D
6assignvariableop_5_extract_features_bn_moving_variance:@5
assignvariableop_6_conv1_kernel:@@+
assignvariableop_7_conv1_bias:@/
!assignvariableop_8_conv1_bn_gamma:@.
 assignvariableop_9_conv1_bn_beta:@6
(assignvariableop_10_conv1_bn_moving_mean:@:
,assignvariableop_11_conv1_bn_moving_variance:@6
 assignvariableop_12_conv2_kernel:@@,
assignvariableop_13_conv2_bias:@0
"assignvariableop_14_conv2_bn_gamma:@/
!assignvariableop_15_conv2_bn_beta:@6
(assignvariableop_16_conv2_bn_moving_mean:@:
,assignvariableop_17_conv2_bn_moving_variance:@6
#assignvariableop_18_d1_dense_kernel:	@0
!assignvariableop_19_d1_dense_bias:	.
assignvariableop_20_d1_bn_gamma:	-
assignvariableop_21_d1_bn_beta:	4
%assignvariableop_22_d1_bn_moving_mean:	8
)assignvariableop_23_d1_bn_moving_variance:	7
#assignvariableop_24_d2_dense_kernel:
0
!assignvariableop_25_d2_dense_bias:	.
assignvariableop_26_d2_bn_gamma:	-
assignvariableop_27_d2_bn_beta:	4
%assignvariableop_28_d2_bn_moving_mean:	8
)assignvariableop_29_d2_bn_moving_variance:	;
(assignvariableop_30_dense_predict_kernel:	4
&assignvariableop_31_dense_predict_bias:$
assignvariableop_32_beta_1: $
assignvariableop_33_beta_2: #
assignvariableop_34_decay: +
!assignvariableop_35_learning_rate: .
$assignvariableop_36_cond_1_adam_iter:	 0
&assignvariableop_37_current_loss_scale: (
assignvariableop_38_good_steps:	 #
assignvariableop_39_total: #
assignvariableop_40_count: &
assignvariableop_41_aucs:$
assignvariableop_42_ns:O
9assignvariableop_43_cond_1_adam_extract_features_kernel_m:@E
7assignvariableop_44_cond_1_adam_extract_features_bias_m:@I
;assignvariableop_45_cond_1_adam_extract_features_bn_gamma_m:@H
:assignvariableop_46_cond_1_adam_extract_features_bn_beta_m:@D
.assignvariableop_47_cond_1_adam_conv1_kernel_m:@@:
,assignvariableop_48_cond_1_adam_conv1_bias_m:@>
0assignvariableop_49_cond_1_adam_conv1_bn_gamma_m:@=
/assignvariableop_50_cond_1_adam_conv1_bn_beta_m:@D
.assignvariableop_51_cond_1_adam_conv2_kernel_m:@@:
,assignvariableop_52_cond_1_adam_conv2_bias_m:@>
0assignvariableop_53_cond_1_adam_conv2_bn_gamma_m:@=
/assignvariableop_54_cond_1_adam_conv2_bn_beta_m:@D
1assignvariableop_55_cond_1_adam_d1_dense_kernel_m:	@>
/assignvariableop_56_cond_1_adam_d1_dense_bias_m:	<
-assignvariableop_57_cond_1_adam_d1_bn_gamma_m:	;
,assignvariableop_58_cond_1_adam_d1_bn_beta_m:	E
1assignvariableop_59_cond_1_adam_d2_dense_kernel_m:
>
/assignvariableop_60_cond_1_adam_d2_dense_bias_m:	<
-assignvariableop_61_cond_1_adam_d2_bn_gamma_m:	;
,assignvariableop_62_cond_1_adam_d2_bn_beta_m:	I
6assignvariableop_63_cond_1_adam_dense_predict_kernel_m:	B
4assignvariableop_64_cond_1_adam_dense_predict_bias_m:O
9assignvariableop_65_cond_1_adam_extract_features_kernel_v:@E
7assignvariableop_66_cond_1_adam_extract_features_bias_v:@I
;assignvariableop_67_cond_1_adam_extract_features_bn_gamma_v:@H
:assignvariableop_68_cond_1_adam_extract_features_bn_beta_v:@D
.assignvariableop_69_cond_1_adam_conv1_kernel_v:@@:
,assignvariableop_70_cond_1_adam_conv1_bias_v:@>
0assignvariableop_71_cond_1_adam_conv1_bn_gamma_v:@=
/assignvariableop_72_cond_1_adam_conv1_bn_beta_v:@D
.assignvariableop_73_cond_1_adam_conv2_kernel_v:@@:
,assignvariableop_74_cond_1_adam_conv2_bias_v:@>
0assignvariableop_75_cond_1_adam_conv2_bn_gamma_v:@=
/assignvariableop_76_cond_1_adam_conv2_bn_beta_v:@D
1assignvariableop_77_cond_1_adam_d1_dense_kernel_v:	@>
/assignvariableop_78_cond_1_adam_d1_dense_bias_v:	<
-assignvariableop_79_cond_1_adam_d1_bn_gamma_v:	;
,assignvariableop_80_cond_1_adam_d1_bn_beta_v:	E
1assignvariableop_81_cond_1_adam_d2_dense_kernel_v:
>
/assignvariableop_82_cond_1_adam_d2_dense_bias_v:	<
-assignvariableop_83_cond_1_adam_d2_bn_gamma_v:	;
,assignvariableop_84_cond_1_adam_d2_bn_beta_v:	I
6assignvariableop_85_cond_1_adam_dense_predict_kernel_v:	B
4assignvariableop_86_cond_1_adam_dense_predict_bias_v:
identity_88¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_9±0
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*×/
valueÍ/BÊ/XB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEBBoptimizer/loss_scale/current_loss_scale/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/loss_scale/good_steps/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB3keras_api/metrics/1/AUCs/.ATTRIBUTES/VARIABLE_VALUEB1keras_api/metrics/1/Ns/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH£
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*Å
value»B¸XB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ù
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ö
_output_shapesã
à::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*f
dtypes\
Z2X		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp(assignvariableop_extract_features_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp(assignvariableop_1_extract_features_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp,assignvariableop_2_extract_features_bn_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp+assignvariableop_3_extract_features_bn_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_4AssignVariableOp2assignvariableop_4_extract_features_bn_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_5AssignVariableOp6assignvariableop_5_extract_features_bn_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv1_bn_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv1_bn_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp(assignvariableop_10_conv1_bn_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp,assignvariableop_11_conv1_bn_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp assignvariableop_12_conv2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_conv2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2_bn_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2_bn_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp(assignvariableop_16_conv2_bn_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp,assignvariableop_17_conv2_bn_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp#assignvariableop_18_d1_dense_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp!assignvariableop_19_d1_dense_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_d1_bn_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_d1_bn_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp%assignvariableop_22_d1_bn_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp)assignvariableop_23_d1_bn_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp#assignvariableop_24_d2_dense_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp!assignvariableop_25_d2_dense_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_d2_bn_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_d2_bn_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp%assignvariableop_28_d2_bn_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp)assignvariableop_29_d2_bn_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_dense_predict_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp&assignvariableop_31_dense_predict_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOpassignvariableop_32_beta_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOpassignvariableop_33_beta_2Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOpassignvariableop_34_decayIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp!assignvariableop_35_learning_rateIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_36AssignVariableOp$assignvariableop_36_cond_1_adam_iterIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp&assignvariableop_37_current_loss_scaleIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_38AssignVariableOpassignvariableop_38_good_stepsIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOpassignvariableop_39_totalIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOpassignvariableop_40_countIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOpassignvariableop_41_aucsIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOpassignvariableop_42_nsIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_43AssignVariableOp9assignvariableop_43_cond_1_adam_extract_features_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_44AssignVariableOp7assignvariableop_44_cond_1_adam_extract_features_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_45AssignVariableOp;assignvariableop_45_cond_1_adam_extract_features_bn_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_46AssignVariableOp:assignvariableop_46_cond_1_adam_extract_features_bn_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp.assignvariableop_47_cond_1_adam_conv1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp,assignvariableop_48_cond_1_adam_conv1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_49AssignVariableOp0assignvariableop_49_cond_1_adam_conv1_bn_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_50AssignVariableOp/assignvariableop_50_cond_1_adam_conv1_bn_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp.assignvariableop_51_cond_1_adam_conv2_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp,assignvariableop_52_cond_1_adam_conv2_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_53AssignVariableOp0assignvariableop_53_cond_1_adam_conv2_bn_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_54AssignVariableOp/assignvariableop_54_cond_1_adam_conv2_bn_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_55AssignVariableOp1assignvariableop_55_cond_1_adam_d1_dense_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_56AssignVariableOp/assignvariableop_56_cond_1_adam_d1_dense_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp-assignvariableop_57_cond_1_adam_d1_bn_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp,assignvariableop_58_cond_1_adam_d1_bn_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_59AssignVariableOp1assignvariableop_59_cond_1_adam_d2_dense_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_60AssignVariableOp/assignvariableop_60_cond_1_adam_d2_dense_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp-assignvariableop_61_cond_1_adam_d2_bn_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp,assignvariableop_62_cond_1_adam_d2_bn_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_63AssignVariableOp6assignvariableop_63_cond_1_adam_dense_predict_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_64AssignVariableOp4assignvariableop_64_cond_1_adam_dense_predict_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_65AssignVariableOp9assignvariableop_65_cond_1_adam_extract_features_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_66AssignVariableOp7assignvariableop_66_cond_1_adam_extract_features_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_67AssignVariableOp;assignvariableop_67_cond_1_adam_extract_features_bn_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_68AssignVariableOp:assignvariableop_68_cond_1_adam_extract_features_bn_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp.assignvariableop_69_cond_1_adam_conv1_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp,assignvariableop_70_cond_1_adam_conv1_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_71AssignVariableOp0assignvariableop_71_cond_1_adam_conv1_bn_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_72AssignVariableOp/assignvariableop_72_cond_1_adam_conv1_bn_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOp.assignvariableop_73_cond_1_adam_conv2_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_74AssignVariableOp,assignvariableop_74_cond_1_adam_conv2_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_75AssignVariableOp0assignvariableop_75_cond_1_adam_conv2_bn_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_76AssignVariableOp/assignvariableop_76_cond_1_adam_conv2_bn_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_77AssignVariableOp1assignvariableop_77_cond_1_adam_d1_dense_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_78AssignVariableOp/assignvariableop_78_cond_1_adam_d1_dense_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_79AssignVariableOp-assignvariableop_79_cond_1_adam_d1_bn_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_80AssignVariableOp,assignvariableop_80_cond_1_adam_d1_bn_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_81AssignVariableOp1assignvariableop_81_cond_1_adam_d2_dense_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_82AssignVariableOp/assignvariableop_82_cond_1_adam_d2_dense_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_83AssignVariableOp-assignvariableop_83_cond_1_adam_d2_bn_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_84AssignVariableOp,assignvariableop_84_cond_1_adam_d2_bn_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_85AssignVariableOp6assignvariableop_85_cond_1_adam_dense_predict_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_86AssignVariableOp4assignvariableop_86_cond_1_adam_dense_predict_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 É
Identity_87Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_88IdentityIdentity_87:output:0^NoOp_1*
T0*
_output_shapes
: ¶
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_88Identity_88:output:0*Å
_input_shapes³
°: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ò
£
D__inference_conv2_BN_layer_call_and_return_conditional_losses_557023

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpb
CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@r
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@q
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
IdentityIdentity
Cast_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 
e
I__inference_conv1_dropout_layer_call_and_return_conditional_losses_559069

inputs
identity[
IdentityIdentityinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò
£
D__inference_conv1_BN_layer_call_and_return_conditional_losses_559143

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpb
CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@r
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@q
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
IdentityIdentity
Cast_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@º
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
J
.__inference_conv1_dropout_layer_call_fn_559055

inputs
identityÄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv1_dropout_layer_call_and_return_conditional_losses_557335m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
î
M
1__inference_combine_features_layer_call_fn_559385

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_combine_features_layer_call_and_return_conditional_losses_557108i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ãÌ
þ
A__inference_model_layer_call_and_return_conditional_losses_558930

inputsR
<extract_features_conv1d_expanddims_1_readvariableop_resource:@>
0extract_features_biasadd_readvariableop_resource:@I
;extract_features_bn_assignmovingavg_readvariableop_resource:@K
=extract_features_bn_assignmovingavg_1_readvariableop_resource:@G
9extract_features_bn_batchnorm_mul_readvariableop_resource:@C
5extract_features_bn_batchnorm_readvariableop_resource:@G
1conv1_conv1d_expanddims_1_readvariableop_resource:@@3
%conv1_biasadd_readvariableop_resource:@>
0conv1_bn_assignmovingavg_readvariableop_resource:@@
2conv1_bn_assignmovingavg_1_readvariableop_resource:@<
.conv1_bn_batchnorm_mul_readvariableop_resource:@8
*conv1_bn_batchnorm_readvariableop_resource:@G
1conv2_conv1d_expanddims_1_readvariableop_resource:@@3
%conv2_biasadd_readvariableop_resource:@>
0conv2_bn_assignmovingavg_readvariableop_resource:@@
2conv2_bn_assignmovingavg_1_readvariableop_resource:@<
.conv2_bn_batchnorm_mul_readvariableop_resource:@8
*conv2_bn_batchnorm_readvariableop_resource:@:
'd1_dense_matmul_readvariableop_resource:	@7
(d1_dense_biasadd_readvariableop_resource:	<
-d1_bn_assignmovingavg_readvariableop_resource:	>
/d1_bn_assignmovingavg_1_readvariableop_resource:	:
+d1_bn_batchnorm_mul_readvariableop_resource:	6
'd1_bn_batchnorm_readvariableop_resource:	;
'd2_dense_matmul_readvariableop_resource:
7
(d2_dense_biasadd_readvariableop_resource:	<
-d2_bn_assignmovingavg_readvariableop_resource:	>
/d2_bn_assignmovingavg_1_readvariableop_resource:	:
+d2_bn_batchnorm_mul_readvariableop_resource:	6
'd2_bn_batchnorm_readvariableop_resource:	?
,dense_predict_matmul_readvariableop_resource:	;
-dense_predict_biasadd_readvariableop_resource:
identity¢conv1/BiasAdd/ReadVariableOp¢(conv1/Conv1D/ExpandDims_1/ReadVariableOp¢conv1_BN/AssignMovingAvg¢'conv1_BN/AssignMovingAvg/ReadVariableOp¢conv1_BN/AssignMovingAvg_1¢)conv1_BN/AssignMovingAvg_1/ReadVariableOp¢!conv1_BN/batchnorm/ReadVariableOp¢%conv1_BN/batchnorm/mul/ReadVariableOp¢conv2/BiasAdd/ReadVariableOp¢(conv2/Conv1D/ExpandDims_1/ReadVariableOp¢conv2_BN/AssignMovingAvg¢'conv2_BN/AssignMovingAvg/ReadVariableOp¢conv2_BN/AssignMovingAvg_1¢)conv2_BN/AssignMovingAvg_1/ReadVariableOp¢!conv2_BN/batchnorm/ReadVariableOp¢%conv2_BN/batchnorm/mul/ReadVariableOp¢d1_BN/AssignMovingAvg¢$d1_BN/AssignMovingAvg/ReadVariableOp¢d1_BN/AssignMovingAvg_1¢&d1_BN/AssignMovingAvg_1/ReadVariableOp¢d1_BN/batchnorm/ReadVariableOp¢"d1_BN/batchnorm/mul/ReadVariableOp¢d1_dense/BiasAdd/ReadVariableOp¢d1_dense/MatMul/ReadVariableOp¢d2_BN/AssignMovingAvg¢$d2_BN/AssignMovingAvg/ReadVariableOp¢d2_BN/AssignMovingAvg_1¢&d2_BN/AssignMovingAvg_1/ReadVariableOp¢d2_BN/batchnorm/ReadVariableOp¢"d2_BN/batchnorm/mul/ReadVariableOp¢d2_dense/BiasAdd/ReadVariableOp¢d2_dense/MatMul/ReadVariableOp¢$dense_predict/BiasAdd/ReadVariableOp¢#dense_predict/MatMul/ReadVariableOp¢'extract_features/BiasAdd/ReadVariableOp¢3extract_features/Conv1D/ExpandDims_1/ReadVariableOp¢#extract_features_BN/AssignMovingAvg¢2extract_features_BN/AssignMovingAvg/ReadVariableOp¢%extract_features_BN/AssignMovingAvg_1¢4extract_features_BN/AssignMovingAvg_1/ReadVariableOp¢,extract_features_BN/batchnorm/ReadVariableOp¢0extract_features_BN/batchnorm/mul/ReadVariableOps
extract_features/CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq
&extract_features/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¿
"extract_features/Conv1D/ExpandDims
ExpandDimsextract_features/Cast:y:0/extract_features/Conv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
3extract_features/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<extract_features_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0ª
)extract_features/Conv1D/ExpandDims_1/CastCast;extract_features/Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@j
(extract_features/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Å
$extract_features/Conv1D/ExpandDims_1
ExpandDims-extract_features/Conv1D/ExpandDims_1/Cast:y:01extract_features/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@é
extract_features/Conv1DConv2D+extract_features/Conv1D/ExpandDims:output:0-extract_features/Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
«
extract_features/Conv1D/SqueezeSqueeze extract_features/Conv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
'extract_features/BiasAdd/ReadVariableOpReadVariableOp0extract_features_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
extract_features/BiasAdd/CastCast/extract_features/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@¯
extract_features/BiasAddBiasAdd(extract_features/Conv1D/Squeeze:output:0!extract_features/BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
extract_features_BN/CastCast!extract_features/BiasAdd:output:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
2extract_features_BN/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Á
 extract_features_BN/moments/meanMeanextract_features_BN/Cast:y:0;extract_features_BN/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(
(extract_features_BN/moments/StopGradientStopGradient)extract_features_BN/moments/mean:output:0*
T0*"
_output_shapes
:@Ò
-extract_features_BN/moments/SquaredDifferenceSquaredDifferenceextract_features_BN/Cast:y:01extract_features_BN/moments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
6extract_features_BN/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Þ
$extract_features_BN/moments/varianceMean1extract_features_BN/moments/SquaredDifference:z:0?extract_features_BN/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(
#extract_features_BN/moments/SqueezeSqueeze)extract_features_BN/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 
%extract_features_BN/moments/Squeeze_1Squeeze-extract_features_BN/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 n
)extract_features_BN/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<ª
2extract_features_BN/AssignMovingAvg/ReadVariableOpReadVariableOp;extract_features_bn_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0½
'extract_features_BN/AssignMovingAvg/subSub:extract_features_BN/AssignMovingAvg/ReadVariableOp:value:0,extract_features_BN/moments/Squeeze:output:0*
T0*
_output_shapes
:@´
'extract_features_BN/AssignMovingAvg/mulMul+extract_features_BN/AssignMovingAvg/sub:z:02extract_features_BN/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@ü
#extract_features_BN/AssignMovingAvgAssignSubVariableOp;extract_features_bn_assignmovingavg_readvariableop_resource+extract_features_BN/AssignMovingAvg/mul:z:03^extract_features_BN/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+extract_features_BN/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<®
4extract_features_BN/AssignMovingAvg_1/ReadVariableOpReadVariableOp=extract_features_bn_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0Ã
)extract_features_BN/AssignMovingAvg_1/subSub<extract_features_BN/AssignMovingAvg_1/ReadVariableOp:value:0.extract_features_BN/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@º
)extract_features_BN/AssignMovingAvg_1/mulMul-extract_features_BN/AssignMovingAvg_1/sub:z:04extract_features_BN/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@
%extract_features_BN/AssignMovingAvg_1AssignSubVariableOp=extract_features_bn_assignmovingavg_1_readvariableop_resource-extract_features_BN/AssignMovingAvg_1/mul:z:05^extract_features_BN/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#extract_features_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:­
!extract_features_BN/batchnorm/addAddV2.extract_features_BN/moments/Squeeze_1:output:0,extract_features_BN/batchnorm/add/y:output:0*
T0*
_output_shapes
:@x
#extract_features_BN/batchnorm/RsqrtRsqrt%extract_features_BN/batchnorm/add:z:0*
T0*
_output_shapes
:@¦
0extract_features_BN/batchnorm/mul/ReadVariableOpReadVariableOp9extract_features_bn_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0°
!extract_features_BN/batchnorm/mulMul'extract_features_BN/batchnorm/Rsqrt:y:08extract_features_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@®
#extract_features_BN/batchnorm/mul_1Mulextract_features_BN/Cast:y:0%extract_features_BN/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¤
#extract_features_BN/batchnorm/mul_2Mul,extract_features_BN/moments/Squeeze:output:0%extract_features_BN/batchnorm/mul:z:0*
T0*
_output_shapes
:@
,extract_features_BN/batchnorm/ReadVariableOpReadVariableOp5extract_features_bn_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0¬
!extract_features_BN/batchnorm/subSub4extract_features_BN/batchnorm/ReadVariableOp:value:0'extract_features_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@»
#extract_features_BN/batchnorm/add_1AddV2'extract_features_BN/batchnorm/mul_1:z:0%extract_features_BN/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
extract_features_BN/Cast_1Cast'extract_features_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
extract_features_RELU/ReluReluextract_features_BN/Cast_1:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¸
conv1/Conv1D/ExpandDims
ExpandDims(extract_features_RELU/Relu:activations:0$conv1/Conv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
(conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
conv1/Conv1D/ExpandDims_1/CastCast0conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@@_
conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¤
conv1/Conv1D/ExpandDims_1
ExpandDims"conv1/Conv1D/ExpandDims_1/Cast:y:0&conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@È
conv1/Conv1DConv2D conv1/Conv1D/ExpandDims:output:0"conv1/Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv1/Conv1D/SqueezeSqueezeconv1/Conv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0t
conv1/BiasAdd/CastCast$conv1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@
conv1/BiasAddBiasAddconv1/Conv1D/Squeeze:output:0conv1/BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@{
conv1_BN/CastCastconv1/BiasAdd:output:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@x
'conv1_BN/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"        
conv1_BN/moments/meanMeanconv1_BN/Cast:y:00conv1_BN/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(z
conv1_BN/moments/StopGradientStopGradientconv1_BN/moments/mean:output:0*
T0*"
_output_shapes
:@±
"conv1_BN/moments/SquaredDifferenceSquaredDifferenceconv1_BN/Cast:y:0&conv1_BN/moments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@|
+conv1_BN/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ½
conv1_BN/moments/varianceMean&conv1_BN/moments/SquaredDifference:z:04conv1_BN/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(
conv1_BN/moments/SqueezeSqueezeconv1_BN/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 
conv1_BN/moments/Squeeze_1Squeeze"conv1_BN/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 c
conv1_BN/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
'conv1_BN/AssignMovingAvg/ReadVariableOpReadVariableOp0conv1_bn_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
conv1_BN/AssignMovingAvg/subSub/conv1_BN/AssignMovingAvg/ReadVariableOp:value:0!conv1_BN/moments/Squeeze:output:0*
T0*
_output_shapes
:@
conv1_BN/AssignMovingAvg/mulMul conv1_BN/AssignMovingAvg/sub:z:0'conv1_BN/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@Ð
conv1_BN/AssignMovingAvgAssignSubVariableOp0conv1_bn_assignmovingavg_readvariableop_resource conv1_BN/AssignMovingAvg/mul:z:0(^conv1_BN/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0e
 conv1_BN/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
)conv1_BN/AssignMovingAvg_1/ReadVariableOpReadVariableOp2conv1_bn_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0¢
conv1_BN/AssignMovingAvg_1/subSub1conv1_BN/AssignMovingAvg_1/ReadVariableOp:value:0#conv1_BN/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@
conv1_BN/AssignMovingAvg_1/mulMul"conv1_BN/AssignMovingAvg_1/sub:z:0)conv1_BN/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@Ø
conv1_BN/AssignMovingAvg_1AssignSubVariableOp2conv1_bn_assignmovingavg_1_readvariableop_resource"conv1_BN/AssignMovingAvg_1/mul:z:0*^conv1_BN/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0]
conv1_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv1_BN/batchnorm/addAddV2#conv1_BN/moments/Squeeze_1:output:0!conv1_BN/batchnorm/add/y:output:0*
T0*
_output_shapes
:@b
conv1_BN/batchnorm/RsqrtRsqrtconv1_BN/batchnorm/add:z:0*
T0*
_output_shapes
:@
%conv1_BN/batchnorm/mul/ReadVariableOpReadVariableOp.conv1_bn_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0
conv1_BN/batchnorm/mulMulconv1_BN/batchnorm/Rsqrt:y:0-conv1_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@
conv1_BN/batchnorm/mul_1Mulconv1_BN/Cast:y:0conv1_BN/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
conv1_BN/batchnorm/mul_2Mul!conv1_BN/moments/Squeeze:output:0conv1_BN/batchnorm/mul:z:0*
T0*
_output_shapes
:@
!conv1_BN/batchnorm/ReadVariableOpReadVariableOp*conv1_bn_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0
conv1_BN/batchnorm/subSub)conv1_BN/batchnorm/ReadVariableOp:value:0conv1_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
conv1_BN/batchnorm/add_1AddV2conv1_BN/batchnorm/mul_1:z:0conv1_BN/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
conv1_BN/Cast_1Castconv1_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@k
conv1_RELU/ReluReluconv1_BN/Cast_1:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Y
conv1_mp/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :¥
conv1_mp/ExpandDims
ExpandDimsconv1_RELU/Relu:activations:0 conv1_mp/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¸
conv1_mp/MaxPoolMaxPoolconv1_mp/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

conv1_mp/SqueezeSqueezeconv1_mp/MaxPool:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
f
conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ©
conv2/Conv1D/ExpandDims
ExpandDimsconv1_mp/Squeeze:output:0$conv2/Conv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
(conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
conv2/Conv1D/ExpandDims_1/CastCast0conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@@_
conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¤
conv2/Conv1D/ExpandDims_1
ExpandDims"conv2/Conv1D/ExpandDims_1/Cast:y:0&conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@È
conv2/Conv1DConv2D conv2/Conv1D/ExpandDims:output:0"conv2/Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv2/Conv1D/SqueezeSqueezeconv2/Conv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ~
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0t
conv2/BiasAdd/CastCast$conv2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@
conv2/BiasAddBiasAddconv2/Conv1D/Squeeze:output:0conv2/BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@{
conv2_BN/CastCastconv2/BiasAdd:output:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@x
'conv2_BN/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"        
conv2_BN/moments/meanMeanconv2_BN/Cast:y:00conv2_BN/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(z
conv2_BN/moments/StopGradientStopGradientconv2_BN/moments/mean:output:0*
T0*"
_output_shapes
:@±
"conv2_BN/moments/SquaredDifferenceSquaredDifferenceconv2_BN/Cast:y:0&conv2_BN/moments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@|
+conv2_BN/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ½
conv2_BN/moments/varianceMean&conv2_BN/moments/SquaredDifference:z:04conv2_BN/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(
conv2_BN/moments/SqueezeSqueezeconv2_BN/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 
conv2_BN/moments/Squeeze_1Squeeze"conv2_BN/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 c
conv2_BN/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
'conv2_BN/AssignMovingAvg/ReadVariableOpReadVariableOp0conv2_bn_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2_BN/AssignMovingAvg/subSub/conv2_BN/AssignMovingAvg/ReadVariableOp:value:0!conv2_BN/moments/Squeeze:output:0*
T0*
_output_shapes
:@
conv2_BN/AssignMovingAvg/mulMul conv2_BN/AssignMovingAvg/sub:z:0'conv2_BN/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@Ð
conv2_BN/AssignMovingAvgAssignSubVariableOp0conv2_bn_assignmovingavg_readvariableop_resource conv2_BN/AssignMovingAvg/mul:z:0(^conv2_BN/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0e
 conv2_BN/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
)conv2_BN/AssignMovingAvg_1/ReadVariableOpReadVariableOp2conv2_bn_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0¢
conv2_BN/AssignMovingAvg_1/subSub1conv2_BN/AssignMovingAvg_1/ReadVariableOp:value:0#conv2_BN/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@
conv2_BN/AssignMovingAvg_1/mulMul"conv2_BN/AssignMovingAvg_1/sub:z:0)conv2_BN/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@Ø
conv2_BN/AssignMovingAvg_1AssignSubVariableOp2conv2_bn_assignmovingavg_1_readvariableop_resource"conv2_BN/AssignMovingAvg_1/mul:z:0*^conv2_BN/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0]
conv2_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2_BN/batchnorm/addAddV2#conv2_BN/moments/Squeeze_1:output:0!conv2_BN/batchnorm/add/y:output:0*
T0*
_output_shapes
:@b
conv2_BN/batchnorm/RsqrtRsqrtconv2_BN/batchnorm/add:z:0*
T0*
_output_shapes
:@
%conv2_BN/batchnorm/mul/ReadVariableOpReadVariableOp.conv2_bn_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2_BN/batchnorm/mulMulconv2_BN/batchnorm/Rsqrt:y:0-conv2_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@
conv2_BN/batchnorm/mul_1Mulconv2_BN/Cast:y:0conv2_BN/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
conv2_BN/batchnorm/mul_2Mul!conv2_BN/moments/Squeeze:output:0conv2_BN/batchnorm/mul:z:0*
T0*
_output_shapes
:@
!conv2_BN/batchnorm/ReadVariableOpReadVariableOp*conv2_bn_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2_BN/batchnorm/subSub)conv2_BN/batchnorm/ReadVariableOp:value:0conv2_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
conv2_BN/batchnorm/add_1AddV2conv2_BN/batchnorm/mul_1:z:0conv2_BN/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
conv2_BN/Cast_1Castconv2_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@k
conv2_RELU/ReluReluconv2_BN/Cast_1:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Y
conv2_mp/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :¥
conv2_mp/ExpandDims
ExpandDimsconv2_RELU/Relu:activations:0 conv2_mp/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¸
conv2_mp/MaxPoolMaxPoolconv2_mp/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

conv2_mp/SqueezeSqueezeconv2_mp/MaxPool:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
h
&combine_features/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
combine_features/MaxMaxconv2_mp/Squeeze:output:0/combine_features/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[
d1_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B jz
d1_dropout/dropout/MulMulcombine_features/Max:output:0!d1_dropout/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@e
d1_dropout/dropout/ShapeShapecombine_features/Max:output:0*
T0*
_output_shapes
:»
/d1_dropout/dropout/random_uniform/RandomUniformRandomUniform!d1_dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype0*

seed{*
seed2d
!d1_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B jædÇ
d1_dropout/dropout/GreaterEqualGreaterEqual8d1_dropout/dropout/random_uniform/RandomUniform:output:0*d1_dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
d1_dropout/dropout/CastCast#d1_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
d1_dropout/dropout/Mul_1Muld1_dropout/dropout/Mul:z:0d1_dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
d1_dense/MatMul/ReadVariableOpReadVariableOp'd1_dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0}
d1_dense/MatMul/CastCast&d1_dense/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	@
d1_dense/MatMulMatMuld1_dropout/dropout/Mul_1:z:0d1_dense/MatMul/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d1_dense/BiasAdd/ReadVariableOpReadVariableOp(d1_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0{
d1_dense/BiasAdd/CastCast'd1_dense/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:
d1_dense/BiasAddBiasAddd1_dense/MatMul:product:0d1_dense/BiasAdd/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

d1_BN/CastCastd1_dense/BiasAdd:output:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
$d1_BN/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
d1_BN/moments/meanMeand1_BN/Cast:y:0-d1_BN/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(q
d1_BN/moments/StopGradientStopGradientd1_BN/moments/mean:output:0*
T0*
_output_shapes
:	
d1_BN/moments/SquaredDifferenceSquaredDifferenced1_BN/Cast:y:0#d1_BN/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
(d1_BN/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ±
d1_BN/moments/varianceMean#d1_BN/moments/SquaredDifference:z:01d1_BN/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(z
d1_BN/moments/SqueezeSqueezed1_BN/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 
d1_BN/moments/Squeeze_1Squeezed1_BN/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 `
d1_BN/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
$d1_BN/AssignMovingAvg/ReadVariableOpReadVariableOp-d1_bn_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
d1_BN/AssignMovingAvg/subSub,d1_BN/AssignMovingAvg/ReadVariableOp:value:0d1_BN/moments/Squeeze:output:0*
T0*
_output_shapes	
:
d1_BN/AssignMovingAvg/mulMuld1_BN/AssignMovingAvg/sub:z:0$d1_BN/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Ä
d1_BN/AssignMovingAvgAssignSubVariableOp-d1_bn_assignmovingavg_readvariableop_resourced1_BN/AssignMovingAvg/mul:z:0%^d1_BN/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0b
d1_BN/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
&d1_BN/AssignMovingAvg_1/ReadVariableOpReadVariableOp/d1_bn_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
d1_BN/AssignMovingAvg_1/subSub.d1_BN/AssignMovingAvg_1/ReadVariableOp:value:0 d1_BN/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
d1_BN/AssignMovingAvg_1/mulMuld1_BN/AssignMovingAvg_1/sub:z:0&d1_BN/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:Ì
d1_BN/AssignMovingAvg_1AssignSubVariableOp/d1_bn_assignmovingavg_1_readvariableop_resourced1_BN/AssignMovingAvg_1/mul:z:0'^d1_BN/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0Z
d1_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
d1_BN/batchnorm/addAddV2 d1_BN/moments/Squeeze_1:output:0d1_BN/batchnorm/add/y:output:0*
T0*
_output_shapes	
:]
d1_BN/batchnorm/RsqrtRsqrtd1_BN/batchnorm/add:z:0*
T0*
_output_shapes	
:
"d1_BN/batchnorm/mul/ReadVariableOpReadVariableOp+d1_bn_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0
d1_BN/batchnorm/mulMuld1_BN/batchnorm/Rsqrt:y:0*d1_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:x
d1_BN/batchnorm/mul_1Muld1_BN/Cast:y:0d1_BN/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
d1_BN/batchnorm/mul_2Muld1_BN/moments/Squeeze:output:0d1_BN/batchnorm/mul:z:0*
T0*
_output_shapes	
:
d1_BN/batchnorm/ReadVariableOpReadVariableOp'd1_bn_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0
d1_BN/batchnorm/subSub&d1_BN/batchnorm/ReadVariableOp:value:0d1_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
d1_BN/batchnorm/add_1AddV2d1_BN/batchnorm/mul_1:z:0d1_BN/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
d1_BN/Cast_1Castd1_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
d1_RELU/ReluRelud1_BN/Cast_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
d2_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
value
B jz
d2_dropout/dropout/MulMuld1_RELU/Relu:activations:0!d2_dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
d2_dropout/dropout/ShapeShaped1_RELU/Relu:activations:0*
T0*
_output_shapes
:¼
/d2_dropout/dropout/random_uniform/RandomUniformRandomUniform!d2_dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*

seed{*
seed2d
!d2_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value
B jædÈ
d2_dropout/dropout/GreaterEqualGreaterEqual8d2_dropout/dropout/random_uniform/RandomUniform:output:0*d2_dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d2_dropout/dropout/CastCast#d2_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d2_dropout/dropout/Mul_1Muld2_dropout/dropout/Mul:z:0d2_dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d2_dense/MatMul/ReadVariableOpReadVariableOp'd2_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0~
d2_dense/MatMul/CastCast&d2_dense/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:

d2_dense/MatMulMatMuld2_dropout/dropout/Mul_1:z:0d2_dense/MatMul/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d2_dense/BiasAdd/ReadVariableOpReadVariableOp(d2_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0{
d2_dense/BiasAdd/CastCast'd2_dense/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:
d2_dense/BiasAddBiasAddd2_dense/MatMul:product:0d2_dense/BiasAdd/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

d2_BN/CastCastd2_dense/BiasAdd:output:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
$d2_BN/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
d2_BN/moments/meanMeand2_BN/Cast:y:0-d2_BN/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(q
d2_BN/moments/StopGradientStopGradientd2_BN/moments/mean:output:0*
T0*
_output_shapes
:	
d2_BN/moments/SquaredDifferenceSquaredDifferenced2_BN/Cast:y:0#d2_BN/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
(d2_BN/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ±
d2_BN/moments/varianceMean#d2_BN/moments/SquaredDifference:z:01d2_BN/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(z
d2_BN/moments/SqueezeSqueezed2_BN/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 
d2_BN/moments/Squeeze_1Squeezed2_BN/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 `
d2_BN/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
$d2_BN/AssignMovingAvg/ReadVariableOpReadVariableOp-d2_bn_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
d2_BN/AssignMovingAvg/subSub,d2_BN/AssignMovingAvg/ReadVariableOp:value:0d2_BN/moments/Squeeze:output:0*
T0*
_output_shapes	
:
d2_BN/AssignMovingAvg/mulMuld2_BN/AssignMovingAvg/sub:z:0$d2_BN/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Ä
d2_BN/AssignMovingAvgAssignSubVariableOp-d2_bn_assignmovingavg_readvariableop_resourced2_BN/AssignMovingAvg/mul:z:0%^d2_BN/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0b
d2_BN/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
&d2_BN/AssignMovingAvg_1/ReadVariableOpReadVariableOp/d2_bn_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
d2_BN/AssignMovingAvg_1/subSub.d2_BN/AssignMovingAvg_1/ReadVariableOp:value:0 d2_BN/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
d2_BN/AssignMovingAvg_1/mulMuld2_BN/AssignMovingAvg_1/sub:z:0&d2_BN/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:Ì
d2_BN/AssignMovingAvg_1AssignSubVariableOp/d2_bn_assignmovingavg_1_readvariableop_resourced2_BN/AssignMovingAvg_1/mul:z:0'^d2_BN/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0Z
d2_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
d2_BN/batchnorm/addAddV2 d2_BN/moments/Squeeze_1:output:0d2_BN/batchnorm/add/y:output:0*
T0*
_output_shapes	
:]
d2_BN/batchnorm/RsqrtRsqrtd2_BN/batchnorm/add:z:0*
T0*
_output_shapes	
:
"d2_BN/batchnorm/mul/ReadVariableOpReadVariableOp+d2_bn_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0
d2_BN/batchnorm/mulMuld2_BN/batchnorm/Rsqrt:y:0*d2_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:x
d2_BN/batchnorm/mul_1Muld2_BN/Cast:y:0d2_BN/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
d2_BN/batchnorm/mul_2Muld2_BN/moments/Squeeze:output:0d2_BN/batchnorm/mul:z:0*
T0*
_output_shapes	
:
d2_BN/batchnorm/ReadVariableOpReadVariableOp'd2_bn_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0
d2_BN/batchnorm/subSub&d2_BN/batchnorm/ReadVariableOp:value:0d2_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
d2_BN/batchnorm/add_1AddV2d2_BN/batchnorm/mul_1:z:0d2_BN/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
d2_BN/Cast_1Castd2_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
d2_RELU/ReluRelud2_BN/Cast_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_predict/CastCastd2_RELU/Relu:activations:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#dense_predict/MatMul/ReadVariableOpReadVariableOp,dense_predict_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_predict/MatMulMatMuldense_predict/Cast:y:0+dense_predict/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$dense_predict/BiasAdd/ReadVariableOpReadVariableOp-dense_predict_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
dense_predict/BiasAddBiasAdddense_predict/MatMul:product:0,dense_predict/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
IdentityIdentitydense_predict/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿí
NoOpNoOp^conv1/BiasAdd/ReadVariableOp)^conv1/Conv1D/ExpandDims_1/ReadVariableOp^conv1_BN/AssignMovingAvg(^conv1_BN/AssignMovingAvg/ReadVariableOp^conv1_BN/AssignMovingAvg_1*^conv1_BN/AssignMovingAvg_1/ReadVariableOp"^conv1_BN/batchnorm/ReadVariableOp&^conv1_BN/batchnorm/mul/ReadVariableOp^conv2/BiasAdd/ReadVariableOp)^conv2/Conv1D/ExpandDims_1/ReadVariableOp^conv2_BN/AssignMovingAvg(^conv2_BN/AssignMovingAvg/ReadVariableOp^conv2_BN/AssignMovingAvg_1*^conv2_BN/AssignMovingAvg_1/ReadVariableOp"^conv2_BN/batchnorm/ReadVariableOp&^conv2_BN/batchnorm/mul/ReadVariableOp^d1_BN/AssignMovingAvg%^d1_BN/AssignMovingAvg/ReadVariableOp^d1_BN/AssignMovingAvg_1'^d1_BN/AssignMovingAvg_1/ReadVariableOp^d1_BN/batchnorm/ReadVariableOp#^d1_BN/batchnorm/mul/ReadVariableOp ^d1_dense/BiasAdd/ReadVariableOp^d1_dense/MatMul/ReadVariableOp^d2_BN/AssignMovingAvg%^d2_BN/AssignMovingAvg/ReadVariableOp^d2_BN/AssignMovingAvg_1'^d2_BN/AssignMovingAvg_1/ReadVariableOp^d2_BN/batchnorm/ReadVariableOp#^d2_BN/batchnorm/mul/ReadVariableOp ^d2_dense/BiasAdd/ReadVariableOp^d2_dense/MatMul/ReadVariableOp%^dense_predict/BiasAdd/ReadVariableOp$^dense_predict/MatMul/ReadVariableOp(^extract_features/BiasAdd/ReadVariableOp4^extract_features/Conv1D/ExpandDims_1/ReadVariableOp$^extract_features_BN/AssignMovingAvg3^extract_features_BN/AssignMovingAvg/ReadVariableOp&^extract_features_BN/AssignMovingAvg_15^extract_features_BN/AssignMovingAvg_1/ReadVariableOp-^extract_features_BN/batchnorm/ReadVariableOp1^extract_features_BN/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2T
(conv1/Conv1D/ExpandDims_1/ReadVariableOp(conv1/Conv1D/ExpandDims_1/ReadVariableOp24
conv1_BN/AssignMovingAvgconv1_BN/AssignMovingAvg2R
'conv1_BN/AssignMovingAvg/ReadVariableOp'conv1_BN/AssignMovingAvg/ReadVariableOp28
conv1_BN/AssignMovingAvg_1conv1_BN/AssignMovingAvg_12V
)conv1_BN/AssignMovingAvg_1/ReadVariableOp)conv1_BN/AssignMovingAvg_1/ReadVariableOp2F
!conv1_BN/batchnorm/ReadVariableOp!conv1_BN/batchnorm/ReadVariableOp2N
%conv1_BN/batchnorm/mul/ReadVariableOp%conv1_BN/batchnorm/mul/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2T
(conv2/Conv1D/ExpandDims_1/ReadVariableOp(conv2/Conv1D/ExpandDims_1/ReadVariableOp24
conv2_BN/AssignMovingAvgconv2_BN/AssignMovingAvg2R
'conv2_BN/AssignMovingAvg/ReadVariableOp'conv2_BN/AssignMovingAvg/ReadVariableOp28
conv2_BN/AssignMovingAvg_1conv2_BN/AssignMovingAvg_12V
)conv2_BN/AssignMovingAvg_1/ReadVariableOp)conv2_BN/AssignMovingAvg_1/ReadVariableOp2F
!conv2_BN/batchnorm/ReadVariableOp!conv2_BN/batchnorm/ReadVariableOp2N
%conv2_BN/batchnorm/mul/ReadVariableOp%conv2_BN/batchnorm/mul/ReadVariableOp2.
d1_BN/AssignMovingAvgd1_BN/AssignMovingAvg2L
$d1_BN/AssignMovingAvg/ReadVariableOp$d1_BN/AssignMovingAvg/ReadVariableOp22
d1_BN/AssignMovingAvg_1d1_BN/AssignMovingAvg_12P
&d1_BN/AssignMovingAvg_1/ReadVariableOp&d1_BN/AssignMovingAvg_1/ReadVariableOp2@
d1_BN/batchnorm/ReadVariableOpd1_BN/batchnorm/ReadVariableOp2H
"d1_BN/batchnorm/mul/ReadVariableOp"d1_BN/batchnorm/mul/ReadVariableOp2B
d1_dense/BiasAdd/ReadVariableOpd1_dense/BiasAdd/ReadVariableOp2@
d1_dense/MatMul/ReadVariableOpd1_dense/MatMul/ReadVariableOp2.
d2_BN/AssignMovingAvgd2_BN/AssignMovingAvg2L
$d2_BN/AssignMovingAvg/ReadVariableOp$d2_BN/AssignMovingAvg/ReadVariableOp22
d2_BN/AssignMovingAvg_1d2_BN/AssignMovingAvg_12P
&d2_BN/AssignMovingAvg_1/ReadVariableOp&d2_BN/AssignMovingAvg_1/ReadVariableOp2@
d2_BN/batchnorm/ReadVariableOpd2_BN/batchnorm/ReadVariableOp2H
"d2_BN/batchnorm/mul/ReadVariableOp"d2_BN/batchnorm/mul/ReadVariableOp2B
d2_dense/BiasAdd/ReadVariableOpd2_dense/BiasAdd/ReadVariableOp2@
d2_dense/MatMul/ReadVariableOpd2_dense/MatMul/ReadVariableOp2L
$dense_predict/BiasAdd/ReadVariableOp$dense_predict/BiasAdd/ReadVariableOp2J
#dense_predict/MatMul/ReadVariableOp#dense_predict/MatMul/ReadVariableOp2R
'extract_features/BiasAdd/ReadVariableOp'extract_features/BiasAdd/ReadVariableOp2j
3extract_features/Conv1D/ExpandDims_1/ReadVariableOp3extract_features/Conv1D/ExpandDims_1/ReadVariableOp2J
#extract_features_BN/AssignMovingAvg#extract_features_BN/AssignMovingAvg2h
2extract_features_BN/AssignMovingAvg/ReadVariableOp2extract_features_BN/AssignMovingAvg/ReadVariableOp2N
%extract_features_BN/AssignMovingAvg_1%extract_features_BN/AssignMovingAvg_12l
4extract_features_BN/AssignMovingAvg_1/ReadVariableOp4extract_features_BN/AssignMovingAvg_1/ReadVariableOp2\
,extract_features_BN/batchnorm/ReadVariableOp,extract_features_BN/batchnorm/ReadVariableOp2d
0extract_features_BN/batchnorm/mul/ReadVariableOp0extract_features_BN/batchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
Á
$__inference_signature_wrapper_558358	
input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@ 

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:	@

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:	

unknown_22:	

unknown_23:


unknown_24:	

unknown_25:	

unknown_26:	

unknown_27:	

unknown_28:	

unknown_29:	

unknown_30:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_556810o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

_user_specified_nameinput
Ù
Ï
4__inference_extract_features_BN_layer_call_fn_558982

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_extract_features_BN_layer_call_and_return_conditional_losses_556885|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¤
A__inference_d2_BN_layer_call_and_return_conditional_losses_559640

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:f
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity
Cast_1:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


L__inference_extract_features_layer_call_and_return_conditional_losses_558956

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0
Conv1D/ExpandDims_1/CastCast*Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Conv1D/ExpandDims_1
ExpandDimsConv1D/ExpandDims_1/Cast:y:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@¶
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0h
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@|
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

g
I__inference_conv1_dropout_layer_call_and_return_conditional_losses_557335

inputs

identity_1[
IdentityIdentityinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ð	
û
I__inference_dense_predict_layer_call_and_return_conditional_losses_559705

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û&
Þ
A__inference_d1_BN_layer_call_and_return_conditional_losses_557186

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:f
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity
Cast_1:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Å
&__inference_d2_BN_layer_call_fn_559618

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_d2_BN_layer_call_and_return_conditional_losses_557272p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ø
D__inference_d2_dense_layer_call_and_return_conditional_losses_559592

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0l
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:
\
MatMulMatMulinputsMatMul/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:i
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï'
è
O__inference_extract_features_BN_layer_call_and_return_conditional_losses_559040

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpb
CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@o
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ¢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@r
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@q
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
IdentityIdentity
Cast_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ù
d
F__inference_d1_dropout_layer_call_and_return_conditional_losses_557452

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ä'
Ý
D__inference_conv1_BN_layer_call_and_return_conditional_losses_556971

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpb
CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@o
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ¢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@r
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@q
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
IdentityIdentity
Cast_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Å
Ä
)__inference_conv2_BN_layer_call_fn_559273

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2_BN_layer_call_and_return_conditional_losses_557023|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

g
I__inference_conv1_dropout_layer_call_and_return_conditional_losses_559065

inputs

identity_1[
IdentityIdentityinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
æÓ
ø
A__inference_model_layer_call_and_return_conditional_losses_558672

inputsR
<extract_features_conv1d_expanddims_1_readvariableop_resource:@>
0extract_features_biasadd_readvariableop_resource:@C
5extract_features_bn_batchnorm_readvariableop_resource:@G
9extract_features_bn_batchnorm_mul_readvariableop_resource:@E
7extract_features_bn_batchnorm_readvariableop_1_resource:@E
7extract_features_bn_batchnorm_readvariableop_2_resource:@G
1conv1_conv1d_expanddims_1_readvariableop_resource:@@3
%conv1_biasadd_readvariableop_resource:@8
*conv1_bn_batchnorm_readvariableop_resource:@<
.conv1_bn_batchnorm_mul_readvariableop_resource:@:
,conv1_bn_batchnorm_readvariableop_1_resource:@:
,conv1_bn_batchnorm_readvariableop_2_resource:@G
1conv2_conv1d_expanddims_1_readvariableop_resource:@@3
%conv2_biasadd_readvariableop_resource:@8
*conv2_bn_batchnorm_readvariableop_resource:@<
.conv2_bn_batchnorm_mul_readvariableop_resource:@:
,conv2_bn_batchnorm_readvariableop_1_resource:@:
,conv2_bn_batchnorm_readvariableop_2_resource:@:
'd1_dense_matmul_readvariableop_resource:	@7
(d1_dense_biasadd_readvariableop_resource:	6
'd1_bn_batchnorm_readvariableop_resource:	:
+d1_bn_batchnorm_mul_readvariableop_resource:	8
)d1_bn_batchnorm_readvariableop_1_resource:	8
)d1_bn_batchnorm_readvariableop_2_resource:	;
'd2_dense_matmul_readvariableop_resource:
7
(d2_dense_biasadd_readvariableop_resource:	6
'd2_bn_batchnorm_readvariableop_resource:	:
+d2_bn_batchnorm_mul_readvariableop_resource:	8
)d2_bn_batchnorm_readvariableop_1_resource:	8
)d2_bn_batchnorm_readvariableop_2_resource:	?
,dense_predict_matmul_readvariableop_resource:	;
-dense_predict_biasadd_readvariableop_resource:
identity¢conv1/BiasAdd/ReadVariableOp¢(conv1/Conv1D/ExpandDims_1/ReadVariableOp¢!conv1_BN/batchnorm/ReadVariableOp¢#conv1_BN/batchnorm/ReadVariableOp_1¢#conv1_BN/batchnorm/ReadVariableOp_2¢%conv1_BN/batchnorm/mul/ReadVariableOp¢conv2/BiasAdd/ReadVariableOp¢(conv2/Conv1D/ExpandDims_1/ReadVariableOp¢!conv2_BN/batchnorm/ReadVariableOp¢#conv2_BN/batchnorm/ReadVariableOp_1¢#conv2_BN/batchnorm/ReadVariableOp_2¢%conv2_BN/batchnorm/mul/ReadVariableOp¢d1_BN/batchnorm/ReadVariableOp¢ d1_BN/batchnorm/ReadVariableOp_1¢ d1_BN/batchnorm/ReadVariableOp_2¢"d1_BN/batchnorm/mul/ReadVariableOp¢d1_dense/BiasAdd/ReadVariableOp¢d1_dense/MatMul/ReadVariableOp¢d2_BN/batchnorm/ReadVariableOp¢ d2_BN/batchnorm/ReadVariableOp_1¢ d2_BN/batchnorm/ReadVariableOp_2¢"d2_BN/batchnorm/mul/ReadVariableOp¢d2_dense/BiasAdd/ReadVariableOp¢d2_dense/MatMul/ReadVariableOp¢$dense_predict/BiasAdd/ReadVariableOp¢#dense_predict/MatMul/ReadVariableOp¢'extract_features/BiasAdd/ReadVariableOp¢3extract_features/Conv1D/ExpandDims_1/ReadVariableOp¢,extract_features_BN/batchnorm/ReadVariableOp¢.extract_features_BN/batchnorm/ReadVariableOp_1¢.extract_features_BN/batchnorm/ReadVariableOp_2¢0extract_features_BN/batchnorm/mul/ReadVariableOps
extract_features/CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq
&extract_features/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¿
"extract_features/Conv1D/ExpandDims
ExpandDimsextract_features/Cast:y:0/extract_features/Conv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ´
3extract_features/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp<extract_features_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype0ª
)extract_features/Conv1D/ExpandDims_1/CastCast;extract_features/Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@j
(extract_features/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Å
$extract_features/Conv1D/ExpandDims_1
ExpandDims-extract_features/Conv1D/ExpandDims_1/Cast:y:01extract_features/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@é
extract_features/Conv1DConv2D+extract_features/Conv1D/ExpandDims:output:0-extract_features/Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
«
extract_features/Conv1D/SqueezeSqueeze extract_features/Conv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ
'extract_features/BiasAdd/ReadVariableOpReadVariableOp0extract_features_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
extract_features/BiasAdd/CastCast/extract_features/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@¯
extract_features/BiasAddBiasAdd(extract_features/Conv1D/Squeeze:output:0!extract_features/BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
extract_features_BN/CastCast!extract_features/BiasAdd:output:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
,extract_features_BN/batchnorm/ReadVariableOpReadVariableOp5extract_features_bn_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0h
#extract_features_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:³
!extract_features_BN/batchnorm/addAddV24extract_features_BN/batchnorm/ReadVariableOp:value:0,extract_features_BN/batchnorm/add/y:output:0*
T0*
_output_shapes
:@x
#extract_features_BN/batchnorm/RsqrtRsqrt%extract_features_BN/batchnorm/add:z:0*
T0*
_output_shapes
:@¦
0extract_features_BN/batchnorm/mul/ReadVariableOpReadVariableOp9extract_features_bn_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0°
!extract_features_BN/batchnorm/mulMul'extract_features_BN/batchnorm/Rsqrt:y:08extract_features_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@®
#extract_features_BN/batchnorm/mul_1Mulextract_features_BN/Cast:y:0%extract_features_BN/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¢
.extract_features_BN/batchnorm/ReadVariableOp_1ReadVariableOp7extract_features_bn_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0®
#extract_features_BN/batchnorm/mul_2Mul6extract_features_BN/batchnorm/ReadVariableOp_1:value:0%extract_features_BN/batchnorm/mul:z:0*
T0*
_output_shapes
:@¢
.extract_features_BN/batchnorm/ReadVariableOp_2ReadVariableOp7extract_features_bn_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0®
!extract_features_BN/batchnorm/subSub6extract_features_BN/batchnorm/ReadVariableOp_2:value:0'extract_features_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@»
#extract_features_BN/batchnorm/add_1AddV2'extract_features_BN/batchnorm/mul_1:z:0%extract_features_BN/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
extract_features_BN/Cast_1Cast'extract_features_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
extract_features_RELU/ReluReluextract_features_BN/Cast_1:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
conv1_dropout/IdentityIdentity(extract_features_RELU/Relu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
conv1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¯
conv1/Conv1D/ExpandDims
ExpandDimsconv1_dropout/Identity:output:0$conv1/Conv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
(conv1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp1conv1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
conv1/Conv1D/ExpandDims_1/CastCast0conv1/Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@@_
conv1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¤
conv1/Conv1D/ExpandDims_1
ExpandDims"conv1/Conv1D/ExpandDims_1/Cast:y:0&conv1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@È
conv1/Conv1DConv2D conv1/Conv1D/ExpandDims:output:0"conv1/Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv1/Conv1D/SqueezeSqueezeconv1/Conv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ~
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0t
conv1/BiasAdd/CastCast$conv1/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@
conv1/BiasAddBiasAddconv1/Conv1D/Squeeze:output:0conv1/BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@{
conv1_BN/CastCastconv1/BiasAdd:output:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
!conv1_BN/batchnorm/ReadVariableOpReadVariableOp*conv1_bn_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0]
conv1_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv1_BN/batchnorm/addAddV2)conv1_BN/batchnorm/ReadVariableOp:value:0!conv1_BN/batchnorm/add/y:output:0*
T0*
_output_shapes
:@b
conv1_BN/batchnorm/RsqrtRsqrtconv1_BN/batchnorm/add:z:0*
T0*
_output_shapes
:@
%conv1_BN/batchnorm/mul/ReadVariableOpReadVariableOp.conv1_bn_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0
conv1_BN/batchnorm/mulMulconv1_BN/batchnorm/Rsqrt:y:0-conv1_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@
conv1_BN/batchnorm/mul_1Mulconv1_BN/Cast:y:0conv1_BN/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
#conv1_BN/batchnorm/ReadVariableOp_1ReadVariableOp,conv1_bn_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0
conv1_BN/batchnorm/mul_2Mul+conv1_BN/batchnorm/ReadVariableOp_1:value:0conv1_BN/batchnorm/mul:z:0*
T0*
_output_shapes
:@
#conv1_BN/batchnorm/ReadVariableOp_2ReadVariableOp,conv1_bn_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0
conv1_BN/batchnorm/subSub+conv1_BN/batchnorm/ReadVariableOp_2:value:0conv1_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
conv1_BN/batchnorm/add_1AddV2conv1_BN/batchnorm/mul_1:z:0conv1_BN/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
conv1_BN/Cast_1Castconv1_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@k
conv1_RELU/ReluReluconv1_BN/Cast_1:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Y
conv1_mp/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :¥
conv1_mp/ExpandDims
ExpandDimsconv1_RELU/Relu:activations:0 conv1_mp/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¸
conv1_mp/MaxPoolMaxPoolconv1_mp/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

conv1_mp/SqueezeSqueezeconv1_mp/MaxPool:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
|
conv2_dropout/IdentityIdentityconv1_mp/Squeeze:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
conv2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ¯
conv2/Conv1D/ExpandDims
ExpandDimsconv2_dropout/Identity:output:0$conv2/Conv1D/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
(conv2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp1conv2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype0
conv2/Conv1D/ExpandDims_1/CastCast0conv2/Conv1D/ExpandDims_1/ReadVariableOp:value:0*

DstT0*

SrcT0*"
_output_shapes
:@@_
conv2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ¤
conv2/Conv1D/ExpandDims_1
ExpandDims"conv2/Conv1D/ExpandDims_1/Cast:y:0&conv2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@È
conv2/Conv1DConv2D conv2/Conv1D/ExpandDims:output:0"conv2/Conv1D/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides

conv2/Conv1D/SqueezeSqueezeconv2/Conv1D:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims

ýÿÿÿÿÿÿÿÿ~
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0t
conv2/BiasAdd/CastCast$conv2/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:@
conv2/BiasAddBiasAddconv2/Conv1D/Squeeze:output:0conv2/BiasAdd/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@{
conv2_BN/CastCastconv2/BiasAdd:output:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
!conv2_BN/batchnorm/ReadVariableOpReadVariableOp*conv2_bn_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0]
conv2_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
conv2_BN/batchnorm/addAddV2)conv2_BN/batchnorm/ReadVariableOp:value:0!conv2_BN/batchnorm/add/y:output:0*
T0*
_output_shapes
:@b
conv2_BN/batchnorm/RsqrtRsqrtconv2_BN/batchnorm/add:z:0*
T0*
_output_shapes
:@
%conv2_BN/batchnorm/mul/ReadVariableOpReadVariableOp.conv2_bn_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2_BN/batchnorm/mulMulconv2_BN/batchnorm/Rsqrt:y:0-conv2_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@
conv2_BN/batchnorm/mul_1Mulconv2_BN/Cast:y:0conv2_BN/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
#conv2_BN/batchnorm/ReadVariableOp_1ReadVariableOp,conv2_bn_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0
conv2_BN/batchnorm/mul_2Mul+conv2_BN/batchnorm/ReadVariableOp_1:value:0conv2_BN/batchnorm/mul:z:0*
T0*
_output_shapes
:@
#conv2_BN/batchnorm/ReadVariableOp_2ReadVariableOp,conv2_bn_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0
conv2_BN/batchnorm/subSub+conv2_BN/batchnorm/ReadVariableOp_2:value:0conv2_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
conv2_BN/batchnorm/add_1AddV2conv2_BN/batchnorm/mul_1:z:0conv2_BN/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
conv2_BN/Cast_1Castconv2_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@k
conv2_RELU/ReluReluconv2_BN/Cast_1:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Y
conv2_mp/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :¥
conv2_mp/ExpandDims
ExpandDimsconv2_RELU/Relu:activations:0 conv2_mp/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¸
conv2_mp/MaxPoolMaxPoolconv2_mp/ExpandDims:output:0*
T0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

conv2_mp/SqueezeSqueezeconv2_mp/MaxPool:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
squeeze_dims
h
&combine_features/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
combine_features/MaxMaxconv2_mp/Squeeze:output:0/combine_features/Max/reduction_indices:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@p
d1_dropout/IdentityIdentitycombine_features/Max:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
d1_dense/MatMul/ReadVariableOpReadVariableOp'd1_dense_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0}
d1_dense/MatMul/CastCast&d1_dense/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	@
d1_dense/MatMulMatMuld1_dropout/Identity:output:0d1_dense/MatMul/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d1_dense/BiasAdd/ReadVariableOpReadVariableOp(d1_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0{
d1_dense/BiasAdd/CastCast'd1_dense/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:
d1_dense/BiasAddBiasAddd1_dense/MatMul:product:0d1_dense/BiasAdd/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

d1_BN/CastCastd1_dense/BiasAdd:output:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d1_BN/batchnorm/ReadVariableOpReadVariableOp'd1_bn_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0Z
d1_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
d1_BN/batchnorm/addAddV2&d1_BN/batchnorm/ReadVariableOp:value:0d1_BN/batchnorm/add/y:output:0*
T0*
_output_shapes	
:]
d1_BN/batchnorm/RsqrtRsqrtd1_BN/batchnorm/add:z:0*
T0*
_output_shapes	
:
"d1_BN/batchnorm/mul/ReadVariableOpReadVariableOp+d1_bn_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0
d1_BN/batchnorm/mulMuld1_BN/batchnorm/Rsqrt:y:0*d1_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:x
d1_BN/batchnorm/mul_1Muld1_BN/Cast:y:0d1_BN/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 d1_BN/batchnorm/ReadVariableOp_1ReadVariableOp)d1_bn_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0
d1_BN/batchnorm/mul_2Mul(d1_BN/batchnorm/ReadVariableOp_1:value:0d1_BN/batchnorm/mul:z:0*
T0*
_output_shapes	
:
 d1_BN/batchnorm/ReadVariableOp_2ReadVariableOp)d1_bn_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0
d1_BN/batchnorm/subSub(d1_BN/batchnorm/ReadVariableOp_2:value:0d1_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
d1_BN/batchnorm/add_1AddV2d1_BN/batchnorm/mul_1:z:0d1_BN/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
d1_BN/Cast_1Castd1_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
d1_RELU/ReluRelud1_BN/Cast_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
d2_dropout/IdentityIdentityd1_RELU/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d2_dense/MatMul/ReadVariableOpReadVariableOp'd2_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0~
d2_dense/MatMul/CastCast&d2_dense/MatMul/ReadVariableOp:value:0*

DstT0*

SrcT0* 
_output_shapes
:

d2_dense/MatMulMatMuld2_dropout/Identity:output:0d2_dense/MatMul/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d2_dense/BiasAdd/ReadVariableOpReadVariableOp(d2_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0{
d2_dense/BiasAdd/CastCast'd2_dense/BiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:
d2_dense/BiasAddBiasAddd2_dense/MatMul:product:0d2_dense/BiasAdd/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

d2_BN/CastCastd2_dense/BiasAdd:output:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
d2_BN/batchnorm/ReadVariableOpReadVariableOp'd2_bn_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0Z
d2_BN/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
d2_BN/batchnorm/addAddV2&d2_BN/batchnorm/ReadVariableOp:value:0d2_BN/batchnorm/add/y:output:0*
T0*
_output_shapes	
:]
d2_BN/batchnorm/RsqrtRsqrtd2_BN/batchnorm/add:z:0*
T0*
_output_shapes	
:
"d2_BN/batchnorm/mul/ReadVariableOpReadVariableOp+d2_bn_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0
d2_BN/batchnorm/mulMuld2_BN/batchnorm/Rsqrt:y:0*d2_BN/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:x
d2_BN/batchnorm/mul_1Muld2_BN/Cast:y:0d2_BN/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 d2_BN/batchnorm/ReadVariableOp_1ReadVariableOp)d2_bn_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0
d2_BN/batchnorm/mul_2Mul(d2_BN/batchnorm/ReadVariableOp_1:value:0d2_BN/batchnorm/mul:z:0*
T0*
_output_shapes	
:
 d2_BN/batchnorm/ReadVariableOp_2ReadVariableOp)d2_bn_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0
d2_BN/batchnorm/subSub(d2_BN/batchnorm/ReadVariableOp_2:value:0d2_BN/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
d2_BN/batchnorm/add_1AddV2d2_BN/batchnorm/mul_1:z:0d2_BN/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
d2_BN/Cast_1Castd2_BN/batchnorm/add_1:z:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
d2_RELU/ReluRelud2_BN/Cast_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dense_predict/CastCastd2_RELU/Relu:activations:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#dense_predict/MatMul/ReadVariableOpReadVariableOp,dense_predict_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_predict/MatMulMatMuldense_predict/Cast:y:0+dense_predict/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$dense_predict/BiasAdd/ReadVariableOpReadVariableOp-dense_predict_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
dense_predict/BiasAddBiasAdddense_predict/MatMul:product:0,dense_predict/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
IdentityIdentitydense_predict/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp^conv1/BiasAdd/ReadVariableOp)^conv1/Conv1D/ExpandDims_1/ReadVariableOp"^conv1_BN/batchnorm/ReadVariableOp$^conv1_BN/batchnorm/ReadVariableOp_1$^conv1_BN/batchnorm/ReadVariableOp_2&^conv1_BN/batchnorm/mul/ReadVariableOp^conv2/BiasAdd/ReadVariableOp)^conv2/Conv1D/ExpandDims_1/ReadVariableOp"^conv2_BN/batchnorm/ReadVariableOp$^conv2_BN/batchnorm/ReadVariableOp_1$^conv2_BN/batchnorm/ReadVariableOp_2&^conv2_BN/batchnorm/mul/ReadVariableOp^d1_BN/batchnorm/ReadVariableOp!^d1_BN/batchnorm/ReadVariableOp_1!^d1_BN/batchnorm/ReadVariableOp_2#^d1_BN/batchnorm/mul/ReadVariableOp ^d1_dense/BiasAdd/ReadVariableOp^d1_dense/MatMul/ReadVariableOp^d2_BN/batchnorm/ReadVariableOp!^d2_BN/batchnorm/ReadVariableOp_1!^d2_BN/batchnorm/ReadVariableOp_2#^d2_BN/batchnorm/mul/ReadVariableOp ^d2_dense/BiasAdd/ReadVariableOp^d2_dense/MatMul/ReadVariableOp%^dense_predict/BiasAdd/ReadVariableOp$^dense_predict/MatMul/ReadVariableOp(^extract_features/BiasAdd/ReadVariableOp4^extract_features/Conv1D/ExpandDims_1/ReadVariableOp-^extract_features_BN/batchnorm/ReadVariableOp/^extract_features_BN/batchnorm/ReadVariableOp_1/^extract_features_BN/batchnorm/ReadVariableOp_21^extract_features_BN/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2T
(conv1/Conv1D/ExpandDims_1/ReadVariableOp(conv1/Conv1D/ExpandDims_1/ReadVariableOp2F
!conv1_BN/batchnorm/ReadVariableOp!conv1_BN/batchnorm/ReadVariableOp2J
#conv1_BN/batchnorm/ReadVariableOp_1#conv1_BN/batchnorm/ReadVariableOp_12J
#conv1_BN/batchnorm/ReadVariableOp_2#conv1_BN/batchnorm/ReadVariableOp_22N
%conv1_BN/batchnorm/mul/ReadVariableOp%conv1_BN/batchnorm/mul/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2T
(conv2/Conv1D/ExpandDims_1/ReadVariableOp(conv2/Conv1D/ExpandDims_1/ReadVariableOp2F
!conv2_BN/batchnorm/ReadVariableOp!conv2_BN/batchnorm/ReadVariableOp2J
#conv2_BN/batchnorm/ReadVariableOp_1#conv2_BN/batchnorm/ReadVariableOp_12J
#conv2_BN/batchnorm/ReadVariableOp_2#conv2_BN/batchnorm/ReadVariableOp_22N
%conv2_BN/batchnorm/mul/ReadVariableOp%conv2_BN/batchnorm/mul/ReadVariableOp2@
d1_BN/batchnorm/ReadVariableOpd1_BN/batchnorm/ReadVariableOp2D
 d1_BN/batchnorm/ReadVariableOp_1 d1_BN/batchnorm/ReadVariableOp_12D
 d1_BN/batchnorm/ReadVariableOp_2 d1_BN/batchnorm/ReadVariableOp_22H
"d1_BN/batchnorm/mul/ReadVariableOp"d1_BN/batchnorm/mul/ReadVariableOp2B
d1_dense/BiasAdd/ReadVariableOpd1_dense/BiasAdd/ReadVariableOp2@
d1_dense/MatMul/ReadVariableOpd1_dense/MatMul/ReadVariableOp2@
d2_BN/batchnorm/ReadVariableOpd2_BN/batchnorm/ReadVariableOp2D
 d2_BN/batchnorm/ReadVariableOp_1 d2_BN/batchnorm/ReadVariableOp_12D
 d2_BN/batchnorm/ReadVariableOp_2 d2_BN/batchnorm/ReadVariableOp_22H
"d2_BN/batchnorm/mul/ReadVariableOp"d2_BN/batchnorm/mul/ReadVariableOp2B
d2_dense/BiasAdd/ReadVariableOpd2_dense/BiasAdd/ReadVariableOp2@
d2_dense/MatMul/ReadVariableOpd2_dense/MatMul/ReadVariableOp2L
$dense_predict/BiasAdd/ReadVariableOp$dense_predict/BiasAdd/ReadVariableOp2J
#dense_predict/MatMul/ReadVariableOp#dense_predict/MatMul/ReadVariableOp2R
'extract_features/BiasAdd/ReadVariableOp'extract_features/BiasAdd/ReadVariableOp2j
3extract_features/Conv1D/ExpandDims_1/ReadVariableOp3extract_features/Conv1D/ExpandDims_1/ReadVariableOp2\
,extract_features_BN/batchnorm/ReadVariableOp,extract_features_BN/batchnorm/ReadVariableOp2`
.extract_features_BN/batchnorm/ReadVariableOp_1.extract_features_BN/batchnorm/ReadVariableOp_12`
.extract_features_BN/batchnorm/ReadVariableOp_2.extract_features_BN/batchnorm/ReadVariableOp_22d
0extract_features_BN/batchnorm/mul/ReadVariableOp0extract_features_BN/batchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä'
Ý
D__inference_conv2_BN_layer_call_and_return_conditional_losses_557072

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpb
CastCastinputs*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@o
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ¢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@r
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@q
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@f
IdentityIdentity
Cast_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨
G
+__inference_d2_dropout_layer_call_fn_559549

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_d2_dropout_layer_call_and_return_conditional_losses_557493a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

÷
D__inference_d1_dense_layer_call_and_return_conditional_losses_557466

inputs1
matmul_readvariableop_resource:	@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0k
MatMul/CastCastMatMul/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes
:	@\
MatMulMatMulinputsMatMul/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0i
BiasAdd/CastCastBiasAdd/ReadVariableOp:value:0*

DstT0*

SrcT0*
_output_shapes	
:i
BiasAddBiasAddMatMul:product:0BiasAdd/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ë
_
C__inference_d1_RELU_layer_call_and_return_conditional_losses_557486

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
b
F__inference_conv2_RELU_layer_call_and_return_conditional_losses_559354

inputs
identityS
ReluReluinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@g
IdentityIdentityRelu:activations:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

Å
&__inference_d2_BN_layer_call_fn_559605

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_d2_BN_layer_call_and_return_conditional_losses_557223p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷

&__inference_conv1_layer_call_fn_559078

inputs
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1_layer_call_and_return_conditional_losses_557354|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Û&
Þ
A__inference_d2_BN_layer_call_and_return_conditional_losses_559676

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOpV
CastCastinputs*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:f
batchnorm/mul_1MulCast:y:0batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
Cast_1Castbatchnorm/add_1:z:0*

DstT0*

SrcT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentity
Cast_1:y:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
`
D__inference_conv1_mp_layer_call_and_return_conditional_losses_556994

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¯
MaxPoolMaxPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É

)__inference_d1_dense_layer_call_fn_559438

inputs
unknown:	@
	unknown_0:	
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_d1_dense_layer_call_and_return_conditional_losses_557466p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¹
serving_default¥
D
input;
serving_default_input:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿA
dense_predict0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¶
Ö
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer-18
layer-19
layer_with_weights-8
layer-20
layer_with_weights-9
layer-21
layer-22
layer_with_weights-10
layer-23
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
Ù__call__
+Ú&call_and_return_all_conditional_losses
Û_default_save_signature
	Üloss"
_tf_keras_network
"
_tf_keras_input_layer
½

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"
_tf_keras_layer
ì
%axis
	&gamma
'beta
(moving_mean
)moving_variance
*	variables
+trainable_variables
,regularization_losses
-	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"
_tf_keras_layer
§
.	variables
/trainable_variables
0regularization_losses
1	keras_api
á__call__
+â&call_and_return_all_conditional_losses"
_tf_keras_layer
§
2	variables
3trainable_variables
4regularization_losses
5	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses"
_tf_keras_layer
½

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
å__call__
+æ&call_and_return_all_conditional_losses"
_tf_keras_layer
ì
<axis
	=gamma
>beta
?moving_mean
@moving_variance
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
ç__call__
+è&call_and_return_all_conditional_losses"
_tf_keras_layer
§
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
é__call__
+ê&call_and_return_all_conditional_losses"
_tf_keras_layer
§
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses"
_tf_keras_layer
§
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
í__call__
+î&call_and_return_all_conditional_losses"
_tf_keras_layer
½

Qkernel
Rbias
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
ï__call__
+ð&call_and_return_all_conditional_losses"
_tf_keras_layer
ì
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
\	variables
]trainable_variables
^regularization_losses
_	keras_api
ñ__call__
+ò&call_and_return_all_conditional_losses"
_tf_keras_layer
§
`	variables
atrainable_variables
bregularization_losses
c	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"
_tf_keras_layer
§
d	variables
etrainable_variables
fregularization_losses
g	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"
_tf_keras_layer
§
h	variables
itrainable_variables
jregularization_losses
k	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses"
_tf_keras_layer
§
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses"
_tf_keras_layer
½

pkernel
qbias
r	variables
strainable_variables
tregularization_losses
u	keras_api
û__call__
+ü&call_and_return_all_conditional_losses"
_tf_keras_layer
ì
vaxis
	wgamma
xbeta
ymoving_mean
zmoving_variance
{	variables
|trainable_variables
}regularization_losses
~	keras_api
ý__call__
+þ&call_and_return_all_conditional_losses"
_tf_keras_layer
ª
	variables
trainable_variables
regularization_losses
	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
	axis

gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
kernel
	bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Â
 
loss_scale
¡base_optimizer
¢beta_1
£beta_2

¤decay
¥learning_rate
	¦iterm­ m®&m¯'m°6m±7m²=m³>m´QmµRm¶Xm·Ym¸pm¹qmºwm»xm¼	m½	m¾	m¿	mÀ	mÁ	mÂvÃ vÄ&vÅ'vÆ6vÇ7vÈ=vÉ>vÊQvËRvÌXvÍYvÎpvÏqvÐwvÑxvÒ	vÓ	vÔ	vÕ	vÖ	v×	vØ"
	optimizer

0
 1
&2
'3
(4
)5
66
77
=8
>9
?10
@11
Q12
R13
X14
Y15
Z16
[17
p18
q19
w20
x21
y22
z23
24
25
26
27
28
29
30
31"
trackable_list_wrapper
Ì
0
 1
&2
'3
64
75
=6
>7
Q8
R9
X10
Y11
p12
q13
w14
x15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
Ó
§non_trainable_variables
¨layers
©metrics
 ªlayer_regularization_losses
«layer_metrics
	variables
trainable_variables
regularization_losses
Ù__call__
Û_default_save_signature
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
-:+@2extract_features/kernel
#:!@2extract_features/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¬non_trainable_variables
­layers
®metrics
 ¯layer_regularization_losses
°layer_metrics
!	variables
"trainable_variables
#regularization_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses
'Þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%@2extract_features_BN/gamma
&:$@2extract_features_BN/beta
/:-@ (2extract_features_BN/moving_mean
3:1@ (2#extract_features_BN/moving_variance
<
&0
'1
(2
)3"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
±non_trainable_variables
²layers
³metrics
 ´layer_regularization_losses
µlayer_metrics
*	variables
+trainable_variables
,regularization_losses
ß__call__
+à&call_and_return_all_conditional_losses
'à"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
.	variables
/trainable_variables
0regularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
2	variables
3trainable_variables
4regularization_losses
ã__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
": @@2conv1/kernel
:@2
conv1/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ànon_trainable_variables
Álayers
Âmetrics
 Ãlayer_regularization_losses
Älayer_metrics
8	variables
9trainable_variables
:regularization_losses
å__call__
+æ&call_and_return_all_conditional_losses
'æ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2conv1_BN/gamma
:@2conv1_BN/beta
$:"@ (2conv1_BN/moving_mean
(:&@ (2conv1_BN/moving_variance
<
=0
>1
?2
@3"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ånon_trainable_variables
Ælayers
Çmetrics
 Èlayer_regularization_losses
Élayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
ç__call__
+è&call_and_return_all_conditional_losses
'è"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ênon_trainable_variables
Ëlayers
Ìmetrics
 Ílayer_regularization_losses
Îlayer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
é__call__
+ê&call_and_return_all_conditional_losses
'ê"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ïnon_trainable_variables
Ðlayers
Ñmetrics
 Òlayer_regularization_losses
Ólayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
ë__call__
+ì&call_and_return_all_conditional_losses
'ì"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
í__call__
+î&call_and_return_all_conditional_losses
'î"call_and_return_conditional_losses"
_generic_user_object
": @@2conv2/kernel
:@2
conv2/bias
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ùnon_trainable_variables
Úlayers
Ûmetrics
 Ülayer_regularization_losses
Ýlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
ï__call__
+ð&call_and_return_all_conditional_losses
'ð"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2conv2_BN/gamma
:@2conv2_BN/beta
$:"@ (2conv2_BN/moving_mean
(:&@ (2conv2_BN/moving_variance
<
X0
Y1
Z2
[3"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
\	variables
]trainable_variables
^regularization_losses
ñ__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ãnon_trainable_variables
älayers
åmetrics
 ælayer_regularization_losses
çlayer_metrics
`	variables
atrainable_variables
bregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ènon_trainable_variables
élayers
êmetrics
 ëlayer_regularization_losses
ìlayer_metrics
d	variables
etrainable_variables
fregularization_losses
õ__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ínon_trainable_variables
îlayers
ïmetrics
 ðlayer_regularization_losses
ñlayer_metrics
h	variables
itrainable_variables
jregularization_losses
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ònon_trainable_variables
ólayers
ômetrics
 õlayer_regularization_losses
ölayer_metrics
l	variables
mtrainable_variables
nregularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
": 	@2d1_dense/kernel
:2d1_dense/bias
.
p0
q1"
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
r	variables
strainable_variables
tregularization_losses
û__call__
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2d1_BN/gamma
:2
d1_BN/beta
":  (2d1_BN/moving_mean
&:$ (2d1_BN/moving_variance
<
w0
x1
y2
z3"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ünon_trainable_variables
ýlayers
þmetrics
 ÿlayer_regularization_losses
layer_metrics
{	variables
|trainable_variables
}regularization_losses
ý__call__
+þ&call_and_return_all_conditional_losses
'þ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
·
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
#:!
2d2_dense/kernel
:2d2_dense/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2d2_BN/gamma
:2
d2_BN/beta
":  (2d2_BN/moving_mean
&:$ (2d2_BN/moving_variance
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
':%	2dense_predict/kernel
 :2dense_predict/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
H
current_loss_scale
 
good_steps"
_generic_user_object
"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2cond_1/Adam/iter
h
(0
)1
?2
@3
Z4
[5
y6
z7
8
9"
trackable_list_wrapper
Ö
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23"
trackable_list_wrapper
0
¡0
¢1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
: 2current_loss_scale
:	 2
good_steps
R

£total

¤count
¥	variables
¦	keras_api"
_tf_keras_metric
m

§drugs
	¨AUCs
©Ns
ª_call_result
«	variables
¬	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
£0
¤1"
trackable_list_wrapper
.
¥	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2AUCs
: (2Ns
 "
trackable_dict_wrapper
0
¨0
©1"
trackable_list_wrapper
.
«	variables"
_generic_user_object
9:7@2%cond_1/Adam/extract_features/kernel/m
/:-@2#cond_1/Adam/extract_features/bias/m
3:1@2'cond_1/Adam/extract_features_BN/gamma/m
2:0@2&cond_1/Adam/extract_features_BN/beta/m
.:,@@2cond_1/Adam/conv1/kernel/m
$:"@2cond_1/Adam/conv1/bias/m
(:&@2cond_1/Adam/conv1_BN/gamma/m
':%@2cond_1/Adam/conv1_BN/beta/m
.:,@@2cond_1/Adam/conv2/kernel/m
$:"@2cond_1/Adam/conv2/bias/m
(:&@2cond_1/Adam/conv2_BN/gamma/m
':%@2cond_1/Adam/conv2_BN/beta/m
.:,	@2cond_1/Adam/d1_dense/kernel/m
(:&2cond_1/Adam/d1_dense/bias/m
&:$2cond_1/Adam/d1_BN/gamma/m
%:#2cond_1/Adam/d1_BN/beta/m
/:-
2cond_1/Adam/d2_dense/kernel/m
(:&2cond_1/Adam/d2_dense/bias/m
&:$2cond_1/Adam/d2_BN/gamma/m
%:#2cond_1/Adam/d2_BN/beta/m
3:1	2"cond_1/Adam/dense_predict/kernel/m
,:*2 cond_1/Adam/dense_predict/bias/m
9:7@2%cond_1/Adam/extract_features/kernel/v
/:-@2#cond_1/Adam/extract_features/bias/v
3:1@2'cond_1/Adam/extract_features_BN/gamma/v
2:0@2&cond_1/Adam/extract_features_BN/beta/v
.:,@@2cond_1/Adam/conv1/kernel/v
$:"@2cond_1/Adam/conv1/bias/v
(:&@2cond_1/Adam/conv1_BN/gamma/v
':%@2cond_1/Adam/conv1_BN/beta/v
.:,@@2cond_1/Adam/conv2/kernel/v
$:"@2cond_1/Adam/conv2/bias/v
(:&@2cond_1/Adam/conv2_BN/gamma/v
':%@2cond_1/Adam/conv2_BN/beta/v
.:,	@2cond_1/Adam/d1_dense/kernel/v
(:&2cond_1/Adam/d1_dense/bias/v
&:$2cond_1/Adam/d1_BN/gamma/v
%:#2cond_1/Adam/d1_BN/beta/v
/:-
2cond_1/Adam/d2_dense/kernel/v
(:&2cond_1/Adam/d2_dense/bias/v
&:$2cond_1/Adam/d2_BN/gamma/v
%:#2cond_1/Adam/d2_BN/beta/v
3:1	2"cond_1/Adam/dense_predict/kernel/v
,:*2 cond_1/Adam/dense_predict/bias/v
æ2ã
&__inference_model_layer_call_fn_557614
&__inference_model_layer_call_fn_558427
&__inference_model_layer_call_fn_558496
&__inference_model_layer_call_fn_558095À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
A__inference_model_layer_call_and_return_conditional_losses_558672
A__inference_model_layer_call_and_return_conditional_losses_558930
A__inference_model_layer_call_and_return_conditional_losses_558188
A__inference_model_layer_call_and_return_conditional_losses_558281À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÊBÇ
!__inference__wrapped_model_556810input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
À2½
__inference_loss_12088¢
²
FullArgSpec
args
jy_true
jy_pred
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_extract_features_layer_call_fn_558939¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_extract_features_layer_call_and_return_conditional_losses_558956¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¦2£
4__inference_extract_features_BN_layer_call_fn_558969
4__inference_extract_features_BN_layer_call_fn_558982´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ü2Ù
O__inference_extract_features_BN_layer_call_and_return_conditional_losses_559004
O__inference_extract_features_BN_layer_call_and_return_conditional_losses_559040´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
6__inference_extract_features_RELU_layer_call_fn_559045¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
û2ø
Q__inference_extract_features_RELU_layer_call_and_return_conditional_losses_559050¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
.__inference_conv1_dropout_layer_call_fn_559055
.__inference_conv1_dropout_layer_call_fn_559060´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
I__inference_conv1_dropout_layer_call_and_return_conditional_losses_559065
I__inference_conv1_dropout_layer_call_and_return_conditional_losses_559069´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
&__inference_conv1_layer_call_fn_559078¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_conv1_layer_call_and_return_conditional_losses_559095¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
)__inference_conv1_BN_layer_call_fn_559108
)__inference_conv1_BN_layer_call_fn_559121´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_conv1_BN_layer_call_and_return_conditional_losses_559143
D__inference_conv1_BN_layer_call_and_return_conditional_losses_559179´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_conv1_RELU_layer_call_fn_559184¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv1_RELU_layer_call_and_return_conditional_losses_559189¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þ2û
)__inference_conv1_mp_layer_call_fn_559194
)__inference_conv1_mp_layer_call_fn_559199¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±
D__inference_conv1_mp_layer_call_and_return_conditional_losses_559207
D__inference_conv1_mp_layer_call_and_return_conditional_losses_559215¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
.__inference_conv2_dropout_layer_call_fn_559220
.__inference_conv2_dropout_layer_call_fn_559225´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
I__inference_conv2_dropout_layer_call_and_return_conditional_losses_559230
I__inference_conv2_dropout_layer_call_and_return_conditional_losses_559234´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
&__inference_conv2_layer_call_fn_559243¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë2è
A__inference_conv2_layer_call_and_return_conditional_losses_559260¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
)__inference_conv2_BN_layer_call_fn_559273
)__inference_conv2_BN_layer_call_fn_559286´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_conv2_BN_layer_call_and_return_conditional_losses_559308
D__inference_conv2_BN_layer_call_and_return_conditional_losses_559344´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_conv2_RELU_layer_call_fn_559349¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2_RELU_layer_call_and_return_conditional_losses_559354¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þ2û
)__inference_conv2_mp_layer_call_fn_559359
)__inference_conv2_mp_layer_call_fn_559364¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±
D__inference_conv2_mp_layer_call_and_return_conditional_losses_559372
D__inference_conv2_mp_layer_call_and_return_conditional_losses_559380¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
1__inference_combine_features_layer_call_fn_559385
1__inference_combine_features_layer_call_fn_559390¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ä2Á
L__inference_combine_features_layer_call_and_return_conditional_losses_559396
L__inference_combine_features_layer_call_and_return_conditional_losses_559402¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
+__inference_d1_dropout_layer_call_fn_559407
+__inference_d1_dropout_layer_call_fn_559412´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_d1_dropout_layer_call_and_return_conditional_losses_559417
F__inference_d1_dropout_layer_call_and_return_conditional_losses_559429´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_d1_dense_layer_call_fn_559438¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_d1_dense_layer_call_and_return_conditional_losses_559450¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
&__inference_d1_BN_layer_call_fn_559463
&__inference_d1_BN_layer_call_fn_559476´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
À2½
A__inference_d1_BN_layer_call_and_return_conditional_losses_559498
A__inference_d1_BN_layer_call_and_return_conditional_losses_559534´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_d1_RELU_layer_call_fn_559539¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_d1_RELU_layer_call_and_return_conditional_losses_559544¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
+__inference_d2_dropout_layer_call_fn_559549
+__inference_d2_dropout_layer_call_fn_559554´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_d2_dropout_layer_call_and_return_conditional_losses_559559
F__inference_d2_dropout_layer_call_and_return_conditional_losses_559571´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_d2_dense_layer_call_fn_559580¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_d2_dense_layer_call_and_return_conditional_losses_559592¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
&__inference_d2_BN_layer_call_fn_559605
&__inference_d2_BN_layer_call_fn_559618´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
À2½
A__inference_d2_BN_layer_call_and_return_conditional_losses_559640
A__inference_d2_BN_layer_call_and_return_conditional_losses_559676´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_d2_RELU_layer_call_fn_559681¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_d2_RELU_layer_call_and_return_conditional_losses_559686¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_dense_predict_layer_call_fn_559695¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_dense_predict_layer_call_and_return_conditional_losses_559705¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÉBÆ
$__inference_signature_wrapper_558358input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
	J
ConstÌ
!__inference__wrapped_model_556810¦( )&('67@=?>QR[XZYpqzwyx;¢8
1¢.
,)
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "=ª:
8
dense_predict'$
dense_predictÿÿÿÿÿÿÿÿÿÇ
L__inference_combine_features_layer_call_and_return_conditional_losses_559396wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 µ
L__inference_combine_features_layer_call_and_return_conditional_losses_559402e<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
1__inference_combine_features_layer_call_fn_559385jE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1__inference_combine_features_layer_call_fn_559390X<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@Ä
D__inference_conv1_BN_layer_call_and_return_conditional_losses_559143|@=?>@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ä
D__inference_conv1_BN_layer_call_and_return_conditional_losses_559179|?@=>@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
)__inference_conv1_BN_layer_call_fn_559108o@=?>@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
)__inference_conv1_BN_layer_call_fn_559121o?@=>@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¼
F__inference_conv1_RELU_layer_call_and_return_conditional_losses_559189r<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
+__inference_conv1_RELU_layer_call_fn_559184e<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ã
I__inference_conv1_dropout_layer_call_and_return_conditional_losses_559065v@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ã
I__inference_conv1_dropout_layer_call_and_return_conditional_losses_559069v@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
.__inference_conv1_dropout_layer_call_fn_559055i@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
.__inference_conv1_dropout_layer_call_fn_559060i@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@»
A__inference_conv1_layer_call_and_return_conditional_losses_559095v67<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
&__inference_conv1_layer_call_fn_559078i67<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Í
D__inference_conv1_mp_layer_call_and_return_conditional_losses_559207E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 º
D__inference_conv1_mp_layer_call_and_return_conditional_losses_559215r<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ¤
)__inference_conv1_mp_layer_call_fn_559194wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
)__inference_conv1_mp_layer_call_fn_559199e<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ä
D__inference_conv2_BN_layer_call_and_return_conditional_losses_559308|[XZY@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ä
D__inference_conv2_BN_layer_call_and_return_conditional_losses_559344|Z[XY@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
)__inference_conv2_BN_layer_call_fn_559273o[XZY@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
)__inference_conv2_BN_layer_call_fn_559286oZ[XY@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¼
F__inference_conv2_RELU_layer_call_and_return_conditional_losses_559354r<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
+__inference_conv2_RELU_layer_call_fn_559349e<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ã
I__inference_conv2_dropout_layer_call_and_return_conditional_losses_559230v@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ã
I__inference_conv2_dropout_layer_call_and_return_conditional_losses_559234v@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
.__inference_conv2_dropout_layer_call_fn_559220i@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
.__inference_conv2_dropout_layer_call_fn_559225i@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@»
A__inference_conv2_layer_call_and_return_conditional_losses_559260vQR<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
&__inference_conv2_layer_call_fn_559243iQR<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Í
D__inference_conv2_mp_layer_call_and_return_conditional_losses_559372E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 º
D__inference_conv2_mp_layer_call_and_return_conditional_losses_559380r<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ¤
)__inference_conv2_mp_layer_call_fn_559359wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
)__inference_conv2_mp_layer_call_fn_559364e<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@©
A__inference_d1_BN_layer_call_and_return_conditional_losses_559498dzwyx4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ©
A__inference_d1_BN_layer_call_and_return_conditional_losses_559534dyzwx4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
&__inference_d1_BN_layer_call_fn_559463Wzwyx4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_d1_BN_layer_call_fn_559476Wyzwx4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¡
C__inference_d1_RELU_layer_call_and_return_conditional_losses_559544Z0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 y
(__inference_d1_RELU_layer_call_fn_559539M0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
D__inference_d1_dense_layer_call_and_return_conditional_losses_559450]pq/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_d1_dense_layer_call_fn_559438Ppq/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_d1_dropout_layer_call_and_return_conditional_losses_559417\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ¦
F__inference_d1_dropout_layer_call_and_return_conditional_losses_559429\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 ~
+__inference_d1_dropout_layer_call_fn_559407O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "ÿÿÿÿÿÿÿÿÿ@~
+__inference_d1_dropout_layer_call_fn_559412O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "ÿÿÿÿÿÿÿÿÿ@­
A__inference_d2_BN_layer_call_and_return_conditional_losses_559640h4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ­
A__inference_d2_BN_layer_call_and_return_conditional_losses_559676h4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
&__inference_d2_BN_layer_call_fn_559605[4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_d2_BN_layer_call_fn_559618[4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¡
C__inference_d2_RELU_layer_call_and_return_conditional_losses_559686Z0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 y
(__inference_d2_RELU_layer_call_fn_559681M0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
D__inference_d2_dense_layer_call_and_return_conditional_losses_559592`0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_d2_dense_layer_call_fn_559580S0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
F__inference_d2_dropout_layer_call_and_return_conditional_losses_559559^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¨
F__inference_d2_dropout_layer_call_and_return_conditional_losses_559571^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_d2_dropout_layer_call_fn_559549Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_d2_dropout_layer_call_fn_559554Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¬
I__inference_dense_predict_layer_call_and_return_conditional_losses_559705_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_predict_layer_call_fn_559695R0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÏ
O__inference_extract_features_BN_layer_call_and_return_conditional_losses_559004|)&('@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ï
O__inference_extract_features_BN_layer_call_and_return_conditional_losses_559040|()&'@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 §
4__inference_extract_features_BN_layer_call_fn_558969o)&('@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@§
4__inference_extract_features_BN_layer_call_fn_558982o()&'@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ç
Q__inference_extract_features_RELU_layer_call_and_return_conditional_losses_559050r<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
6__inference_extract_features_RELU_layer_call_fn_559045e<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Æ
L__inference_extract_features_layer_call_and_return_conditional_losses_558956v <¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
1__inference_extract_features_layer_call_fn_558939i <¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
__inference_loss_12088mQ¢N
G¢D
 
y_trueÿÿÿÿÿÿÿÿÿ
 
y_predÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÜ
A__inference_model_layer_call_and_return_conditional_losses_558188( )&('67@=?>QR[XZYpqzwyxC¢@
9¢6
,)
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ü
A__inference_model_layer_call_and_return_conditional_losses_558281( ()&'67?@=>QRZ[XYpqyzwxC¢@
9¢6
,)
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ý
A__inference_model_layer_call_and_return_conditional_losses_558672( )&('67@=?>QR[XZYpqzwyxD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ý
A__inference_model_layer_call_and_return_conditional_losses_558930( ()&'67?@=>QRZ[XYpqyzwxD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
&__inference_model_layer_call_fn_557614( )&('67@=?>QR[XZYpqzwyxC¢@
9¢6
,)
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ´
&__inference_model_layer_call_fn_558095( ()&'67?@=>QRZ[XYpqyzwxC¢@
9¢6
,)
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿµ
&__inference_model_layer_call_fn_558427( )&('67@=?>QR[XZYpqzwyxD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿµ
&__inference_model_layer_call_fn_558496( ()&'67?@=>QRZ[XYpqyzwxD¢A
:¢7
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿØ
$__inference_signature_wrapper_558358¯( )&('67@=?>QR[XZYpqzwyxD¢A
¢ 
:ª7
5
input,)
inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"=ª:
8
dense_predict'$
dense_predictÿÿÿÿÿÿÿÿÿ