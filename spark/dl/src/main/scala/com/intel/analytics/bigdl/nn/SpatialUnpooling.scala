/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, T, Table}
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect._
import scala.reflect.runtime.universe

/**
 * TODO: Update documentation
 * @param kW              kernel width
 * @param kH              kernel height
 * @param dW              step size in width
 * @param dH              step size in height
 * @param padW            padding of input in width
 * @param padH            padding of input in height
 * @param format          DataFormat.NCHW or DataFormat.NHWC, indicating the input
 *                        data format
 * TODO: Add optional outputsize
 */

class SpatialUnpooling[T: ClassTag](
  val kW: Int, val kH: Int, val dW: Int, val dH: Int, val padW: Int = 0, val padH: Int = 0,
  val format: DataFormat = DataFormat.NCHW)(implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  var ceilMode = false
  val indices = Tensor[T]()

  def this(kW: Int, kH: Int)(implicit ev: TensorNumeric[T]) {
    this(kW, kH, kW, kH, format = DataFormat.NCHW)
  }

  override def updateOutput(input: Table): Tensor[T] = {
    /* Usage: Input is a table [input, indices] */
    /* TODO: Add support for NHWC */
    /* TODO: Add support for single batch input (3 dimensions) */
    val x : Tensor[T] = input(1)
    require(/*input.dim() == 3 ||*/ x.dim() == 4,
      "SpatialUnpooling: " + ErrorInfo.constrainInputAs3DOrBatch)

    indices.set(input(2))
    val (dimh, dimw, dimc) = format.getHWCDims(x.dim())

    val nInputPlane = x.size(dimc)
    val inputHeight = x.size(dimh)
    val inputWidth = x.size(dimw)

    /* Cannot infer SAME padding from output */
    require(padW >=0 && padH>=0,
      "Can't infer SAME padding from output, pad dimensions must be >= 0" +
      s"pad size(${padW},${padH})")
    require(kW / 2 >= padW && kH / 2 >= padH, "pad should be smaller than half of kernel size" +
      s"pad size($padW,$padH)" +
      s"kernel size($kW, $kH)")

    val oWidth = (inputWidth - 1) * dW - 2*padW + kW
    val oHeight = (inputHeight - 1) * dH - 2*padH + kH
    val nBatch = x.size(1)
    
    // Currently only support NCHW
    output.resize(Array(nBatch, nInputPlane, oHeight, oWidth)).zero()
    if (classTag[T] == classTag[Double]){
        Engine.model.invokeAndWait(
          (1 to nBatch).map(i => () => {
            val curInput = x(i)
            val curOutput = output(i)
            val curIndices = indices(i)
            unpoolingForwardDouble(
              curInput.asInstanceOf[Tensor[Double]],
              curOutput.asInstanceOf[Tensor[Double]],
              curIndices.asInstanceOf[Tensor[Double]],
              oWidth, oHeight
            )
          }
          )
        )
    } else if (classTag[T] == classTag[Float]) {
        Engine.model.invokeAndWait(
          (1 to nBatch).map(i => () => {
            val curInput = x(i)
            val curOutput = output(i)
            val curIndices = indices(i)
            unpoolingForwardFloat(
              curInput.asInstanceOf[Tensor[Float]],
              curOutput.asInstanceOf[Tensor[Float]],
              curIndices.asInstanceOf[Tensor[Float]],
              oWidth, oHeight
            )
          }
          )
        )
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val valuesGrad = gradInput.getOrElse[Tensor[Float]](1, Tensor[Float]()).resizeAs(input[Tensor[T]](1)).zero()
    val indicesGrad = gradInput.getOrElse[Tensor[Float]](2, Tensor[Float]()).resizeAs(indices).zero()
     
    val (dimh, dimw, dimc) = format.getHWCDims(input[Tensor[T]](1).dim())
    val oHeight: Int = gradOutput.size(dimh)
    val oWidth: Int = gradOutput.size(dimw)
       
    val nBatch = input[Tensor[T]](1).size(1)
    if (classTag[T] == classTag[Double]){
        Engine.model.invokeAndWait(
          (1 to nBatch).map(i => () => {
            val curInput = valuesGrad(i)
            val curOutput = gradOutput(i)
            val curIndices = indices(i)
            unpoolingBackwardDouble(
              curInput.asInstanceOf[Tensor[Double]],
              curOutput.asInstanceOf[Tensor[Double]],
              curIndices.asInstanceOf[Tensor[Double]],
              oWidth, oHeight
            )
          }
          )
        )
    } else if (classTag[T] == classTag[Float]) {
        Engine.model.invokeAndWait(
          (1 to nBatch).map(i => () => {
            val curInput = valuesGrad(i)
            val curOutput = gradOutput(i)
            val curIndices = indices(i)
            unpoolingBackwardFloat(
              curInput.asInstanceOf[Tensor[Float]],
              curOutput.asInstanceOf[Tensor[Float]],
              curIndices.asInstanceOf[Tensor[Float]],
              oWidth, oHeight
            )
          }
          )
        )
    }
           
    gradInput(1) = valuesGrad
    gradInput(2) = indicesGrad
    gradInput
  }

  /*
  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialMaxPoolingWithIndices[T]]) {
      return false
    }
    val other = obj.asInstanceOf[SpatialMaxPoolingWithIndices[T]]
    if (this.eq(other)) {
      return true
    }

    kW == other.kW &&
      kH == other.kH &&
      dW == other.dW &&
      dH == other.dH &&
      padW == other.padW &&
      padH == other.padH &&
      indices == other.indices
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + kW.hashCode()
    hash = hash * seed + kH.hashCode()
    hash = hash * seed + dW.hashCode()
    hash = hash * seed + dH.hashCode()
    hash = hash * seed + padW.hashCode()
    hash = hash * seed + padH.hashCode()
    hash = hash * seed + ceilMode.hashCode()
    hash = hash * seed + indices.hashCode()

    hash
  }

  override def toString(): String = {
    s"${getPrintName}($kW, $kH, $dW, $dH, $padW, $padH)"
  }

  override def clearState(): this.type = {
    super.clearState()
    indices.set()
    this
  }*/
    
    private def unpoolingForwardFloat(
        inputTensor: Tensor[Float],
        outputTensor: Tensor[Float],
        indicesTensor: Tensor[Float],
        oWidth: Int, oHeight: Int): Unit = {

        val nSlices = inputTensor.size(1)
        val iHeight = inputTensor.size(2)
        val iWidth = inputTensor.size(3)

        val input = inputTensor.storage().array()
        val inputOffset = inputTensor.storageOffset() - 1
        val output = outputTensor.storage().array()
        val outputOffset = outputTensor.storageOffset() - 1
        val indices = indicesTensor.storage().array()
        val indicesOffset = indicesTensor.storageOffset() - 1

        val slices = Range(0, nSlices).iterator
        while (slices.hasNext) {
          val k = slices.next()
          var i = 0
          while (i < iHeight) {
            var j = 0
            while (j < iWidth) {
              val maxp = indices(i * iWidth + j + indicesOffset + k * iWidth * iHeight).toInt - 1
              output(maxp + k * oWidth * oHeight + outputOffset) =
                input(inputOffset + k * iWidth * iHeight + i * iWidth + j)
              j += 1
            }
            i += 1
          }
        }
    }

    private def unpoolingForwardDouble(
        inputTensor: Tensor[Double],
        outputTensor: Tensor[Double],
        indicesTensor: Tensor[Double],
        oWidth: Int, oHeight: Int): Unit = {

        val nSlices = inputTensor.size(1)
        val iHeight = inputTensor.size(2)
        val iWidth = inputTensor.size(3)

        val input = inputTensor.storage().array()
        val inputOffset = inputTensor.storageOffset() - 1
        val output = outputTensor.storage().array()
        val outputOffset = outputTensor.storageOffset() - 1
        val indices = indicesTensor.storage().array()
        val indicesOffset = indicesTensor.storageOffset() - 1

        val slices = Range(0, nSlices).iterator
        while (slices.hasNext) {
          val k = slices.next()
          var i = 0
          while (i < iHeight) {
            var j = 0
            while (j < iWidth) {
              val maxp = indices(i * iWidth + j + indicesOffset + k * iWidth * iHeight).toInt - 1
              output(maxp + k * oWidth * oHeight + outputOffset) =
                input(inputOffset + k * iWidth * iHeight + i * iWidth + j)
              j += 1
            }
            i += 1
          }
        }
    }
    
    private def unpoolingBackwardFloat(
        gradInputTensor: Tensor[Float],
        gradOutputTensor: Tensor[Float],
        indicesTensor: Tensor[Float],
        oWidth: Int, oHeight: Int): Unit = {

        val nSlices = gradInputTensor.size(1)
        val iHeight = gradInputTensor.size(2)
        val iWidth = gradInputTensor.size(3)

        val input = gradInputTensor.storage().array()
        val inputOffset = gradInputTensor.storageOffset() - 1
        val output = gradOutputTensor.storage().array()
        val outputOffset = gradOutputTensor.storageOffset() - 1
        val indices = indicesTensor.storage().array()
        val indicesOffset = indicesTensor.storageOffset() - 1

        val slices = Range(0, nSlices).iterator
        while (slices.hasNext) {
          val k = slices.next()
          var i = 0
          while (i < iHeight) {
            var j = 0
            while (j < iWidth) {
              val maxp = indices(i * iWidth + j + indicesOffset + k * iWidth * iHeight).toInt - 1
              input(inputOffset + k * iWidth * iHeight + i * iWidth + j) += output(maxp + k * oWidth * oHeight + outputOffset)
              j += 1
            }
            i += 1
          }
        }
    }
    
    private def unpoolingBackwardDouble(
        gradInputTensor: Tensor[Double],
        gradOutputTensor: Tensor[Double],
        indicesTensor: Tensor[Double],
        oWidth: Int, oHeight: Int): Unit = {

        val nSlices = gradInputTensor.size(1)
        val iHeight = gradInputTensor.size(2)
        val iWidth = gradInputTensor.size(3)

        val input = gradInputTensor.storage().array()
        val inputOffset = gradInputTensor.storageOffset() - 1
        val output = gradOutputTensor.storage().array()
        val outputOffset = gradOutputTensor.storageOffset() - 1
        val indices = indicesTensor.storage().array()
        val indicesOffset = indicesTensor.storageOffset() - 1

        val slices = Range(0, nSlices).iterator
        while (slices.hasNext) {
          val k = slices.next()
          var i = 0
          while (i < iHeight) {
            var j = 0
            while (j < iWidth) {
              val maxp = indices(i * iWidth + j + indicesOffset + k * iWidth * iHeight).toInt - 1
              input(inputOffset + k * iWidth * iHeight + i * iWidth + j) += output(maxp + k * oWidth * oHeight + outputOffset)
              j += 1
            }
            i += 1
          }
        }
    }
}

object SpatialUnpooling extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](
      kW: Int,
      kH: Int,
      dW: Int = 1,
      dH: Int = 1,
      padW: Int = 0,
      padH: Int = 0,
      format: DataFormat = DataFormat.NCHW)
      (implicit ev: TensorNumeric[T]): SpatialUnpooling[T] = {
    new SpatialUnpooling[T](kW, kH, dW, dH, padW, padH, format)
  }

  /*
  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val unpooling = super.doLoadModule(context)
    val attrMap = context.bigdlModule.getAttrMap
    val ceil_mode = DataConverter.
      getAttributeValue(context, attrMap.get("ceil_mode")).
      asInstanceOf[Boolean]
    if (ceil_mode) {
      maxPooling.asInstanceOf[SpatialMaxPoolingWithIndices[T]].ceil()
    }
    maxPooling
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              maxPoolingBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {

    super.doSerializeModule(context, maxPoolingBuilder)
    val maxPooling = context.moduleData.module.asInstanceOf[SpatialMaxPoolingWithIndices[T]]
    val ceilBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, ceilBuilder,
      maxPooling.ceilMode, universe.typeOf[Boolean])
    maxPoolingBuilder.putAttr("ceil_mode", ceilBuilder.build)

  }
  */
}
