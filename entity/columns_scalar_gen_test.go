// Code generated by go generate; DO NOT EDIT
// This file is generated by go genrated at 2022-01-25 14:11:22.832059835 +0800 CST m=+0.002903820

package entity

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/chensanle/milvus-sdk-go/v2/internal/proto/schema"
	"github.com/stretchr/testify/assert"
)

func TestColumnBool(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	columnName := fmt.Sprintf("column_Bool_%d", rand.Int())
	columnLen := 8 + rand.Intn(10)

	v := make([]bool, columnLen)
	column := NewColumnBool(columnName, v)

	t.Run("test meta", func(t *testing.T) {
		ft := FieldTypeBool
		assert.Equal(t, "Bool", ft.Name())
		assert.Equal(t, "bool", ft.String())
		pbName, pbType := ft.PbFieldType()
		assert.Equal(t, "Bool", pbName)
		assert.Equal(t, "bool", pbType)
	})

	t.Run("test column attribute", func(t *testing.T) {
		assert.Equal(t, columnName, column.Name())
		assert.Equal(t, FieldTypeBool, column.Type())
		assert.Equal(t, columnLen, column.Len())
		assert.EqualValues(t, v, column.Data())
	})

	t.Run("test column field data", func(t *testing.T) {
		fd := column.FieldData()
		assert.NotNil(t, fd)
		assert.Equal(t, fd.GetFieldName(), columnName)
	})

	t.Run("test column value by idx", func(t *testing.T) {
		_, err := column.ValueByIdx(-1)
		assert.NotNil(t, err)
		_, err = column.ValueByIdx(columnLen)
		assert.NotNil(t, err)
		for i := 0; i < columnLen; i++ {
			v, err := column.ValueByIdx(i)
			assert.Nil(t, err)
			assert.Equal(t, column.values[i], v)
		}
	})
}

func TestFieldDataBoolColumn(t *testing.T) {
	len := rand.Intn(10) + 8
	name := fmt.Sprintf("fd_Bool_%d", rand.Int())
	fd := &schema.FieldData{
		Type:      schema.DataType_Bool,
		FieldName: name,
	}

	t.Run("normal usage", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_BoolData{
					BoolData: &schema.BoolArray{
						Data: make([]bool, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, len)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeBool, column.Type())

		var ev bool
		err = column.AppendValue(ev)
		assert.Equal(t, len+1, column.Len())
		assert.Nil(t, err)

		err = column.AppendValue(struct{}{})
		assert.Equal(t, len+1, column.Len())
		assert.NotNil(t, err)
	})

	t.Run("nil data", func(t *testing.T) {
		fd.Field = nil
		_, err := FieldDataColumn(fd, 0, len)
		assert.NotNil(t, err)
	})

	t.Run("get all data", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_BoolData{
					BoolData: &schema.BoolArray{
						Data: make([]bool, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, -1)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeBool, column.Type())
	})
}

func TestColumnInt8(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	columnName := fmt.Sprintf("column_Int8_%d", rand.Int())
	columnLen := 8 + rand.Intn(10)

	v := make([]int8, columnLen)
	column := NewColumnInt8(columnName, v)

	t.Run("test meta", func(t *testing.T) {
		ft := FieldTypeInt8
		assert.Equal(t, "Int8", ft.Name())
		assert.Equal(t, "int8", ft.String())
		pbName, pbType := ft.PbFieldType()
		assert.Equal(t, "Int", pbName)
		assert.Equal(t, "int32", pbType)
	})

	t.Run("test column attribute", func(t *testing.T) {
		assert.Equal(t, columnName, column.Name())
		assert.Equal(t, FieldTypeInt8, column.Type())
		assert.Equal(t, columnLen, column.Len())
		assert.EqualValues(t, v, column.Data())
	})

	t.Run("test column field data", func(t *testing.T) {
		fd := column.FieldData()
		assert.NotNil(t, fd)
		assert.Equal(t, fd.GetFieldName(), columnName)
	})

	t.Run("test column value by idx", func(t *testing.T) {
		_, err := column.ValueByIdx(-1)
		assert.NotNil(t, err)
		_, err = column.ValueByIdx(columnLen)
		assert.NotNil(t, err)
		for i := 0; i < columnLen; i++ {
			v, err := column.ValueByIdx(i)
			assert.Nil(t, err)
			assert.Equal(t, column.values[i], v)
		}
	})
}

func TestFieldDataInt8Column(t *testing.T) {
	len := rand.Intn(10) + 8
	name := fmt.Sprintf("fd_Int8_%d", rand.Int())
	fd := &schema.FieldData{
		Type:      schema.DataType_Int8,
		FieldName: name,
	}

	t.Run("normal usage", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_IntData{
					IntData: &schema.IntArray{
						Data: make([]int32, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, len)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeInt8, column.Type())

		var ev int8
		err = column.AppendValue(ev)
		assert.Equal(t, len+1, column.Len())
		assert.Nil(t, err)

		err = column.AppendValue(struct{}{})
		assert.Equal(t, len+1, column.Len())
		assert.NotNil(t, err)
	})

	t.Run("nil data", func(t *testing.T) {
		fd.Field = nil
		_, err := FieldDataColumn(fd, 0, len)
		assert.NotNil(t, err)
	})

	t.Run("get all data", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_IntData{
					IntData: &schema.IntArray{
						Data: make([]int32, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, -1)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeInt8, column.Type())
	})
}

func TestColumnInt16(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	columnName := fmt.Sprintf("column_Int16_%d", rand.Int())
	columnLen := 8 + rand.Intn(10)

	v := make([]int16, columnLen)
	column := NewColumnInt16(columnName, v)

	t.Run("test meta", func(t *testing.T) {
		ft := FieldTypeInt16
		assert.Equal(t, "Int16", ft.Name())
		assert.Equal(t, "int16", ft.String())
		pbName, pbType := ft.PbFieldType()
		assert.Equal(t, "Int", pbName)
		assert.Equal(t, "int32", pbType)
	})

	t.Run("test column attribute", func(t *testing.T) {
		assert.Equal(t, columnName, column.Name())
		assert.Equal(t, FieldTypeInt16, column.Type())
		assert.Equal(t, columnLen, column.Len())
		assert.EqualValues(t, v, column.Data())
	})

	t.Run("test column field data", func(t *testing.T) {
		fd := column.FieldData()
		assert.NotNil(t, fd)
		assert.Equal(t, fd.GetFieldName(), columnName)
	})

	t.Run("test column value by idx", func(t *testing.T) {
		_, err := column.ValueByIdx(-1)
		assert.NotNil(t, err)
		_, err = column.ValueByIdx(columnLen)
		assert.NotNil(t, err)
		for i := 0; i < columnLen; i++ {
			v, err := column.ValueByIdx(i)
			assert.Nil(t, err)
			assert.Equal(t, column.values[i], v)
		}
	})
}

func TestFieldDataInt16Column(t *testing.T) {
	len := rand.Intn(10) + 8
	name := fmt.Sprintf("fd_Int16_%d", rand.Int())
	fd := &schema.FieldData{
		Type:      schema.DataType_Int16,
		FieldName: name,
	}

	t.Run("normal usage", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_IntData{
					IntData: &schema.IntArray{
						Data: make([]int32, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, len)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeInt16, column.Type())

		var ev int16
		err = column.AppendValue(ev)
		assert.Equal(t, len+1, column.Len())
		assert.Nil(t, err)

		err = column.AppendValue(struct{}{})
		assert.Equal(t, len+1, column.Len())
		assert.NotNil(t, err)
	})

	t.Run("nil data", func(t *testing.T) {
		fd.Field = nil
		_, err := FieldDataColumn(fd, 0, len)
		assert.NotNil(t, err)
	})

	t.Run("get all data", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_IntData{
					IntData: &schema.IntArray{
						Data: make([]int32, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, -1)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeInt16, column.Type())
	})
}

func TestColumnInt32(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	columnName := fmt.Sprintf("column_Int32_%d", rand.Int())
	columnLen := 8 + rand.Intn(10)

	v := make([]int32, columnLen)
	column := NewColumnInt32(columnName, v)

	t.Run("test meta", func(t *testing.T) {
		ft := FieldTypeInt32
		assert.Equal(t, "Int32", ft.Name())
		assert.Equal(t, "int32", ft.String())
		pbName, pbType := ft.PbFieldType()
		assert.Equal(t, "Int", pbName)
		assert.Equal(t, "int32", pbType)
	})

	t.Run("test column attribute", func(t *testing.T) {
		assert.Equal(t, columnName, column.Name())
		assert.Equal(t, FieldTypeInt32, column.Type())
		assert.Equal(t, columnLen, column.Len())
		assert.EqualValues(t, v, column.Data())
	})

	t.Run("test column field data", func(t *testing.T) {
		fd := column.FieldData()
		assert.NotNil(t, fd)
		assert.Equal(t, fd.GetFieldName(), columnName)
	})

	t.Run("test column value by idx", func(t *testing.T) {
		_, err := column.ValueByIdx(-1)
		assert.NotNil(t, err)
		_, err = column.ValueByIdx(columnLen)
		assert.NotNil(t, err)
		for i := 0; i < columnLen; i++ {
			v, err := column.ValueByIdx(i)
			assert.Nil(t, err)
			assert.Equal(t, column.values[i], v)
		}
	})
}

func TestFieldDataInt32Column(t *testing.T) {
	len := rand.Intn(10) + 8
	name := fmt.Sprintf("fd_Int32_%d", rand.Int())
	fd := &schema.FieldData{
		Type:      schema.DataType_Int32,
		FieldName: name,
	}

	t.Run("normal usage", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_IntData{
					IntData: &schema.IntArray{
						Data: make([]int32, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, len)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeInt32, column.Type())

		var ev int32
		err = column.AppendValue(ev)
		assert.Equal(t, len+1, column.Len())
		assert.Nil(t, err)

		err = column.AppendValue(struct{}{})
		assert.Equal(t, len+1, column.Len())
		assert.NotNil(t, err)
	})

	t.Run("nil data", func(t *testing.T) {
		fd.Field = nil
		_, err := FieldDataColumn(fd, 0, len)
		assert.NotNil(t, err)
	})

	t.Run("get all data", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_IntData{
					IntData: &schema.IntArray{
						Data: make([]int32, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, -1)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeInt32, column.Type())
	})
}

func TestColumnInt64(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	columnName := fmt.Sprintf("column_Int64_%d", rand.Int())
	columnLen := 8 + rand.Intn(10)

	v := make([]int64, columnLen)
	column := NewColumnInt64(columnName, v)

	t.Run("test meta", func(t *testing.T) {
		ft := FieldTypeInt64
		assert.Equal(t, "Int64", ft.Name())
		assert.Equal(t, "int64", ft.String())
		pbName, pbType := ft.PbFieldType()
		assert.Equal(t, "Long", pbName)
		assert.Equal(t, "int64", pbType)
	})

	t.Run("test column attribute", func(t *testing.T) {
		assert.Equal(t, columnName, column.Name())
		assert.Equal(t, FieldTypeInt64, column.Type())
		assert.Equal(t, columnLen, column.Len())
		assert.EqualValues(t, v, column.Data())
	})

	t.Run("test column field data", func(t *testing.T) {
		fd := column.FieldData()
		assert.NotNil(t, fd)
		assert.Equal(t, fd.GetFieldName(), columnName)
	})

	t.Run("test column value by idx", func(t *testing.T) {
		_, err := column.ValueByIdx(-1)
		assert.NotNil(t, err)
		_, err = column.ValueByIdx(columnLen)
		assert.NotNil(t, err)
		for i := 0; i < columnLen; i++ {
			v, err := column.ValueByIdx(i)
			assert.Nil(t, err)
			assert.Equal(t, column.values[i], v)
		}
	})
}

func TestFieldDataInt64Column(t *testing.T) {
	len := rand.Intn(10) + 8
	name := fmt.Sprintf("fd_Int64_%d", rand.Int())
	fd := &schema.FieldData{
		Type:      schema.DataType_Int64,
		FieldName: name,
	}

	t.Run("normal usage", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_LongData{
					LongData: &schema.LongArray{
						Data: make([]int64, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, len)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeInt64, column.Type())

		var ev int64
		err = column.AppendValue(ev)
		assert.Equal(t, len+1, column.Len())
		assert.Nil(t, err)

		err = column.AppendValue(struct{}{})
		assert.Equal(t, len+1, column.Len())
		assert.NotNil(t, err)
	})

	t.Run("nil data", func(t *testing.T) {
		fd.Field = nil
		_, err := FieldDataColumn(fd, 0, len)
		assert.NotNil(t, err)
	})

	t.Run("get all data", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_LongData{
					LongData: &schema.LongArray{
						Data: make([]int64, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, -1)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeInt64, column.Type())
	})
}

func TestColumnFloat(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	columnName := fmt.Sprintf("column_Float_%d", rand.Int())
	columnLen := 8 + rand.Intn(10)

	v := make([]float32, columnLen)
	column := NewColumnFloat(columnName, v)

	t.Run("test meta", func(t *testing.T) {
		ft := FieldTypeFloat
		assert.Equal(t, "Float", ft.Name())
		assert.Equal(t, "float32", ft.String())
		pbName, pbType := ft.PbFieldType()
		assert.Equal(t, "Float", pbName)
		assert.Equal(t, "float32", pbType)
	})

	t.Run("test column attribute", func(t *testing.T) {
		assert.Equal(t, columnName, column.Name())
		assert.Equal(t, FieldTypeFloat, column.Type())
		assert.Equal(t, columnLen, column.Len())
		assert.EqualValues(t, v, column.Data())
	})

	t.Run("test column field data", func(t *testing.T) {
		fd := column.FieldData()
		assert.NotNil(t, fd)
		assert.Equal(t, fd.GetFieldName(), columnName)
	})

	t.Run("test column value by idx", func(t *testing.T) {
		_, err := column.ValueByIdx(-1)
		assert.NotNil(t, err)
		_, err = column.ValueByIdx(columnLen)
		assert.NotNil(t, err)
		for i := 0; i < columnLen; i++ {
			v, err := column.ValueByIdx(i)
			assert.Nil(t, err)
			assert.Equal(t, column.values[i], v)
		}
	})
}

func TestFieldDataFloatColumn(t *testing.T) {
	len := rand.Intn(10) + 8
	name := fmt.Sprintf("fd_Float_%d", rand.Int())
	fd := &schema.FieldData{
		Type:      schema.DataType_Float,
		FieldName: name,
	}

	t.Run("normal usage", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_FloatData{
					FloatData: &schema.FloatArray{
						Data: make([]float32, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, len)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeFloat, column.Type())

		var ev float32
		err = column.AppendValue(ev)
		assert.Equal(t, len+1, column.Len())
		assert.Nil(t, err)

		err = column.AppendValue(struct{}{})
		assert.Equal(t, len+1, column.Len())
		assert.NotNil(t, err)
	})

	t.Run("nil data", func(t *testing.T) {
		fd.Field = nil
		_, err := FieldDataColumn(fd, 0, len)
		assert.NotNil(t, err)
	})

	t.Run("get all data", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_FloatData{
					FloatData: &schema.FloatArray{
						Data: make([]float32, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, -1)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeFloat, column.Type())
	})
}

func TestColumnDouble(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	columnName := fmt.Sprintf("column_Double_%d", rand.Int())
	columnLen := 8 + rand.Intn(10)

	v := make([]float64, columnLen)
	column := NewColumnDouble(columnName, v)

	t.Run("test meta", func(t *testing.T) {
		ft := FieldTypeDouble
		assert.Equal(t, "Double", ft.Name())
		assert.Equal(t, "float64", ft.String())
		pbName, pbType := ft.PbFieldType()
		assert.Equal(t, "Double", pbName)
		assert.Equal(t, "float64", pbType)
	})

	t.Run("test column attribute", func(t *testing.T) {
		assert.Equal(t, columnName, column.Name())
		assert.Equal(t, FieldTypeDouble, column.Type())
		assert.Equal(t, columnLen, column.Len())
		assert.EqualValues(t, v, column.Data())
	})

	t.Run("test column field data", func(t *testing.T) {
		fd := column.FieldData()
		assert.NotNil(t, fd)
		assert.Equal(t, fd.GetFieldName(), columnName)
	})

	t.Run("test column value by idx", func(t *testing.T) {
		_, err := column.ValueByIdx(-1)
		assert.NotNil(t, err)
		_, err = column.ValueByIdx(columnLen)
		assert.NotNil(t, err)
		for i := 0; i < columnLen; i++ {
			v, err := column.ValueByIdx(i)
			assert.Nil(t, err)
			assert.Equal(t, column.values[i], v)
		}
	})
}

func TestFieldDataDoubleColumn(t *testing.T) {
	len := rand.Intn(10) + 8
	name := fmt.Sprintf("fd_Double_%d", rand.Int())
	fd := &schema.FieldData{
		Type:      schema.DataType_Double,
		FieldName: name,
	}

	t.Run("normal usage", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_DoubleData{
					DoubleData: &schema.DoubleArray{
						Data: make([]float64, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, len)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeDouble, column.Type())

		var ev float64
		err = column.AppendValue(ev)
		assert.Equal(t, len+1, column.Len())
		assert.Nil(t, err)

		err = column.AppendValue(struct{}{})
		assert.Equal(t, len+1, column.Len())
		assert.NotNil(t, err)
	})

	t.Run("nil data", func(t *testing.T) {
		fd.Field = nil
		_, err := FieldDataColumn(fd, 0, len)
		assert.NotNil(t, err)
	})

	t.Run("get all data", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_DoubleData{
					DoubleData: &schema.DoubleArray{
						Data: make([]float64, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, -1)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeDouble, column.Type())
	})
}

func TestColumnString(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	columnName := fmt.Sprintf("column_String_%d", rand.Int())
	columnLen := 8 + rand.Intn(10)

	v := make([]string, columnLen)
	column := NewColumnString(columnName, v)

	t.Run("test meta", func(t *testing.T) {
		ft := FieldTypeString
		assert.Equal(t, "String", ft.Name())
		assert.Equal(t, "string", ft.String())
		pbName, pbType := ft.PbFieldType()
		assert.Equal(t, "String", pbName)
		assert.Equal(t, "string", pbType)
	})

	t.Run("test column attribute", func(t *testing.T) {
		assert.Equal(t, columnName, column.Name())
		assert.Equal(t, FieldTypeString, column.Type())
		assert.Equal(t, columnLen, column.Len())
		assert.EqualValues(t, v, column.Data())
	})

	t.Run("test column field data", func(t *testing.T) {
		fd := column.FieldData()
		assert.NotNil(t, fd)
		assert.Equal(t, fd.GetFieldName(), columnName)
	})

	t.Run("test column value by idx", func(t *testing.T) {
		_, err := column.ValueByIdx(-1)
		assert.NotNil(t, err)
		_, err = column.ValueByIdx(columnLen)
		assert.NotNil(t, err)
		for i := 0; i < columnLen; i++ {
			v, err := column.ValueByIdx(i)
			assert.Nil(t, err)
			assert.Equal(t, column.values[i], v)
		}
	})
}

func TestFieldDataStringColumn(t *testing.T) {
	len := rand.Intn(10) + 8
	name := fmt.Sprintf("fd_String_%d", rand.Int())
	fd := &schema.FieldData{
		Type:      schema.DataType_String,
		FieldName: name,
	}

	t.Run("normal usage", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_StringData{
					StringData: &schema.StringArray{
						Data: make([]string, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, len)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeString, column.Type())

		var ev string
		err = column.AppendValue(ev)
		assert.Equal(t, len+1, column.Len())
		assert.Nil(t, err)

		err = column.AppendValue(struct{}{})
		assert.Equal(t, len+1, column.Len())
		assert.NotNil(t, err)
	})

	t.Run("nil data", func(t *testing.T) {
		fd.Field = nil
		_, err := FieldDataColumn(fd, 0, len)
		assert.NotNil(t, err)
	})

	t.Run("get all data", func(t *testing.T) {
		fd.Field = &schema.FieldData_Scalars{
			Scalars: &schema.ScalarField{
				Data: &schema.ScalarField_StringData{
					StringData: &schema.StringArray{
						Data: make([]string, len),
					},
				},
			},
		}
		column, err := FieldDataColumn(fd, 0, -1)
		assert.Nil(t, err)
		assert.NotNil(t, column)

		assert.Equal(t, name, column.Name())
		assert.Equal(t, len, column.Len())
		assert.Equal(t, FieldTypeString, column.Type())
	})
}
