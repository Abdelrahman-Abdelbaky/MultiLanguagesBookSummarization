using E_Book_Summarization.Data;
using GP.FullProject.DTOS;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;
using RestSharp;
using System.IO;
using System.IO;
using System.Text;


namespace GP.FullProject.Controllers
{
    [Authorize]
    public class EnglishBookController : Controller
    {
        private string[] _allowedExtention = new[] { ".pdf" };

        private readonly ApplicationDbContext _dbcontext;
        private readonly Microsoft.AspNetCore.Hosting.IHostingEnvironment _env;
        private string _Path;

        public EnglishBookController(ApplicationDbContext context, Microsoft.AspNetCore.Hosting.IHostingEnvironment env)
        {
            _dbcontext = context;
            _env = env;
        }

        public IActionResult EForm()
        {
            return View();
        }

        public IActionResult summarize_en_book()
        {
            return View();
        }
        public IActionResult loadingEnglish()
        {
            return View();
        }


        public async Task<string> UploadFile(IFormFile file)
        {
            string FPath = Guid.NewGuid() + file.FileName;
            var filePath = Path.Combine(_env.ContentRootPath, @"wwwroot/EBook", FPath);
            using var fileStream = new FileStream(filePath, FileMode.Create);
            await file.CopyToAsync(fileStream);
            return FPath;
        }


        [HttpPost]
        public async Task<IActionResult> EForm([FromForm] BookDto bookDto)
        {
            try
            {


                if (!_allowedExtention.Contains(Path.GetExtension(bookDto.book.FileName)))
                {

                    ViewBag.Message = "the book didnt come from types pdf";

                    return BadRequest("the book didnt come from types pdf");
                }

                string filePath = await UploadFile(bookDto.book);
                using var DataStreem = new MemoryStream();
                await bookDto.book.CopyToAsync(DataStreem);
                int percentage = bookDto.percentage;
               

                var FullPath = $"..\\wwwroot\\EBook\\{filePath}";
                var options = new RestClientOptions("http://127.0.0.1:9000");
                options.MaxTimeout = 10_200_000;
                var client = new RestClient(options);
                var request = new RestRequest($"/uploadfile?data={FullPath}&percantage={percentage}", Method.Post);
                request.AlwaysMultipartFormData = true;
                RestResponse response = await client.ExecuteAsync(request);
                Console.WriteLine(response.Content);
                if (response is not null)
                {
                    var output = JsonConvert.DeserializeObject<List<SumDto>>(response.Content);
                     
                    Console.WriteLine(output);
                    ViewBag.ESUM = output;
                    var filePathsum = Guid.NewGuid();

                    var ebook = new EBook()
                    {
                        BookName = bookDto.BookName,
                        Content = DataStreem.ToArray(),
                        percentage = bookDto.percentage,
                       
                    };

                    await _dbcontext.eBooks.AddAsync(ebook);
                    _dbcontext.SaveChanges();



                    if (output.Count > 0)
                    {
                        foreach (var item in output)
                        {
                            var Esum = new Esum()
                            {
                                sum = item.sum,
                                EBookId = ebook.Id

                            };
                            await _dbcontext.esums.AddAsync(Esum);
                        };
                        _dbcontext.SaveChanges();
                        using (StreamWriter sw = System.IO.File.CreateText(@"wwwroot\EBookSum\" + filePathsum))
                        {
                            sw.WriteLine($"book name : {bookDto.BookName}");
                            foreach (var item in output)
                            {
                                sw.WriteLine(item.chapter);
                                sw.WriteLine(item.sum);
                            }
                            sw.Close();
                        }

                        ViewBag.EfilePath = filePathsum;
                    }
                }
                else
                {
                    return BadRequest("ERROR ");
                }

                return View("summarize_en_book");
            }
            catch (Exception ex) {
                return BadRequest("ERROR ");
            }
        }

        [HttpGet]
        public IActionResult DownloadFile(string path)
        {
             return File(System.IO.File.ReadAllBytes(@"wwwroot\EBookSum\" + path), "text/plain", "downloaded_file.txt"); ;
        }
    }


}
