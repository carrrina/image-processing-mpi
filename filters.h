const float K_smooth[3][3] = 
{
	{1, 1, 1},
	{1, 1, 1},
	{1, 1, 1}
};
const float K_blur[3][3] =
{
	{1, 2, 1},
	{2, 4, 2},
	{1, 2, 1}
};
const float K_sharpen[3][3] = 
{
	{0, -2, 0},
	{-2, 11, -2},
	{0, -2, 0}
};
const float K_mean[3][3] = 
{
	{-1, -1, -1},
	{-1, 9, -1},
	{-1, -1, -1}
};
const float K_emboss[3][3] =
{
	{0, 1, 0},
	{0, 0, 0},
	{0, -1, 0}
};