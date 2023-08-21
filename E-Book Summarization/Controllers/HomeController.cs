using BookSummarization.Services;
using BookSummarization.ViewModel;
using E_Book_Summarization.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;
using RestSharp;
using System.Diagnostics;

namespace E_Book_Summarization.Controllers
{
    [Authorize]
    public class HomeController : Controller
    {
        private readonly IFileServices fileServices;
        private readonly List<string> AllowedExtensions = new List<string> { ".txt" };
        private string downloadedFilePath = string.Empty;
        public HomeController(IFileServices fileServices)
        {
            this.fileServices = fileServices;
        }

        public IActionResult AboutUs()
        {
            return View();
        }
        public IActionResult summarize_ar_book( )
        {
            
            return View();
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> summarize_ar_book(IFormFile file)
        {
            if (AllowedExtensions.Contains(Path.GetExtension(file.FileName).ToLower()))
            {
                try
                {
                    string filePath = await fileServices.UploadFile(file);
                    if (!string.IsNullOrEmpty(filePath))
                    {
                        #region RestSharp
                        var options = new RestClientOptions("http://127.0.0.1:8000");
                        options.MaxTimeout = 36_000_000; //10hs
                        var client = new RestClient(options);
                        var request = new RestRequest("/uploadfile", Method.Post);
                        request.AlwaysMultipartFormData = true;
                        request.AddFile("data", filePath);

                        ViewBag.Message = "File Upload Successful";

                        RestResponse response = await client.ExecuteAsync(request);

                        var output = JsonConvert.DeserializeObject<FileUpload>(response.Content);

                        ViewBag.File = filePath;
                        if (output.Summarization.Count > 0)
                        {
                            fileServices.InsertToFile(output.Summarization);
                            ViewBag.success = "Summary is generating Successfully";
                            ViewBag.summary = output.Summarization;

                        }
                        else
                            ViewBag.FileNotEmpty = "File is Empty";

                        #endregion
                    }
                    else
                    {
                        ViewBag.Message = "File Upload Failed";
                    }
                }
                catch (Exception ex)
                {
                    //Log ex
                    ViewBag.Message = "File Upload Failed from EX";
                }
            }
            else
            {
                ViewBag.Message = "Allowed extension is .txt only";
            }
            return View();
        }

        [HttpGet]
        public IActionResult DownloadFile()
        {
            var memory = fileServices.DownloadFile("OutSummary.txt");
            return File(memory.ToArray(), "text/plain", "Summary.txt");
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}